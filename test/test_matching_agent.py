"""
tests/test_matching_agent.py
=============================
Tests the refactored agent's MCP integration and state machine. The
LLM call inside the "match" node is swapped for a deterministic fake
(see FakeStructuredModel below) so these tests need no API key and no
network access -- they're checking the graph wiring and MCP plumbing,
not the quality of the LLM's judgment.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matching_agent as ma
from matching_agent import MatchAssessment


class FakeStructuredModel:
    """Deterministic stand-in for `model.with_structured_output(MatchAssessment)`.

    Routes on the RESUME portion of the prompt only (not the job
    description, which happens to mention AWS/LangGraph itself) so the
    test actually exercises whether the pipeline differentiates
    candidates instead of just returning a fixed score.
    """

    async def ainvoke(self, prompt: str) -> MatchAssessment:
        resume_text = prompt.split("RESUME (", 1)[1].lower()
        if "langgraph" in resume_text and "aws" in resume_text:
            return MatchAssessment(
                score=93, summary="Excellent match: strong backend plus hands-on MCP/LangGraph.",
                strengths=["AWS", "PostgreSQL", "LangGraph/MCP exposure"], gaps=[],
            )
        if "mongodb" in resume_text:
            return MatchAssessment(
                score=55, summary="Some backend experience but lighter on distributed systems.",
                strengths=["Node.js", "REST APIs"], gaps=["No direct cloud deployment", "No LangGraph/MCP"],
            )
        return MatchAssessment(
            score=20, summary="Primarily frontend background, weak fit for a backend role.",
            strengths=["React", "TypeScript"], gaps=["No backend or distributed-systems experience"],
        )


class FakeModel:
    def with_structured_output(self, schema):
        return FakeStructuredModel()


@pytest.fixture()
def agent_env(tmp_path, monkeypatch):
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "strong_fit.txt").write_text(
        "Priya Sharma\n6 years backend. AWS, PostgreSQL, Docker. Built a tool with LangGraph and MCP."
    )
    (resumes / "partial_fit.txt").write_text(
        "Daniel Okafor\n3 years backend. Node.js, Express, MongoDB. No direct cloud deployment experience."
    )
    (resumes / "weak_fit.txt").write_text(
        "Lena Fischer\n7 years frontend. React, TypeScript, Redux. No backend or database experience."
    )
    results = tmp_path / "results"

    monkeypatch.setenv("RESUME_DIRECTORY", str(resumes))
    monkeypatch.setenv("RESULTS_DIRECTORY", str(results))
    monkeypatch.setenv("WATCH_STATE_FILE", str(resumes / ".mcp_watch_state.json"))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(results / "notifications.log"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")

    for mod in ("filesystem_mcp_server", "notifications_mcp_server"):
        sys.modules.pop(mod, None)

    return resumes, results


@pytest_asyncio.fixture()
async def agent(agent_env):
    a = ma.MatchingAgent(model_name="anthropic:claude-sonnet-5")
    a.model = FakeModel()  # no API key / network needed
    await a.setup()
    return a


@pytest.mark.asyncio
async def test_setup_discovers_tools_from_both_mcp_servers(agent):
    names = set(agent._tool_map)
    # from filesystem_mcp_server.py
    assert {"list_resumes", "read_resume", "watch_directory", "batch_process", "save_match_result"} <= names
    # from notifications_mcp_server.py -- proves multi-server discovery works
    assert {"send_match_notification", "list_notifications"} <= names


@pytest.mark.asyncio
async def test_full_run_ranks_and_notifies_correctly(agent, agent_env):
    resumes, results = agent_env
    jd = Path(ma.HERE / "sample_data" / "job_description.txt").read_text()

    state = await agent.run(jd, "BACKEND-2026-114", str(resumes))

    ranked = [m["candidate_file"] for m in state["matches"]]
    assert ranked == ["strong_fit.txt", "partial_fit.txt", "weak_fit.txt"], (
        "expected matches sorted strongest-first"
    )
    assert state["matches"][0]["score"] == 93
    assert state["notified"] == ["strong_fit.txt"], "only the >=70 scorer should be notified"

    # save_match_result actually persisted, via the MCP tool, not a shortcut
    saved = (results / "match_results.jsonl").read_text().splitlines()
    assert len(saved) == 3

    # send_match_notification actually persisted, via the *second* MCP server
    notified_log = (results / "notifications.log").read_text()
    assert "strong_fit.txt" in notified_log
    assert "weak_fit.txt" not in notified_log


@pytest.mark.asyncio
async def test_second_run_with_no_new_files_short_circuits(agent, agent_env):
    resumes, _ = agent_env
    jd = Path(ma.HERE / "sample_data" / "job_description.txt").read_text()

    first = await agent.run(jd, "BACKEND-2026-114", str(resumes))
    assert len(first["matches"]) == 3

    second = await agent.run(jd, "BACKEND-2026-114", str(resumes))
    assert second.get("matches") is None, (
        "with nothing new, the graph should route straight to END "
        "after check_new_resumes and never reach the match node"
    )


@pytest.mark.asyncio
async def test_watch_then_new_resume_is_picked_up_incrementally(agent, agent_env):
    """Simulates the --watch CLI mode: run once, drop a new resume in,
    run again -- only the new file should get scored/saved, not a
    re-processing of the whole directory.
    """
    resumes, results = agent_env
    jd = Path(ma.HERE / "sample_data" / "job_description.txt").read_text()

    await agent.run(jd, "BACKEND-2026-114", str(resumes))
    (resumes / "late_arrival.txt").write_text(
        "New Candidate\n5 years backend. AWS, PostgreSQL. Has used LangGraph and MCP before."
    )

    second = await agent.run(jd, "BACKEND-2026-114", str(resumes))
    assert [m["candidate_file"] for m in second["matches"]] == ["late_arrival.txt"]

    saved = [json.loads(line) for line in (results / "match_results.jsonl").read_text().splitlines()]
    assert len(saved) == 4  # 3 from the first run + 1 new one, not a full 4-file reprocess

def test_coerce_assessment_accepts_xml_item_lists():
    raw = {
        "score": 88,
        "summary": "Strong overall fit.",
        "strengths": "\n<item>8 years of professional frontend experience</item>\n<item>Strong React/TypeScript leadership</item>\n",
        "gaps": "\n<item>No backend engineering experience</item>\n<item>Limited system design exposure</item>\n",
    }
    assessment = ma._coerce_assessment(raw)
    assert assessment.strengths == [
        "8 years of professional frontend experience",
        "Strong React/TypeScript leadership",
    ]
    assert assessment.gaps == [
        "No backend engineering experience",
        "Limited system design exposure",
    ]


def test_coerce_assessment_accepts_missing_gaps_field():
    raw = {
        "score": 65,
        "summary": "Solid fit overall.",
        "strengths": "<item>Strong project ownership</item>\n<item>Good stakeholder communication</item>",
    }
    assessment = ma._coerce_assessment(raw)
    assert assessment.strengths == ["Strong project ownership", "Good stakeholder communication"]
    assert assessment.gaps == []


def test_reset_watch_removes_state_file(tmp_path, monkeypatch):
    """The CLI reset option allows a completed inbox to be rescanned."""
    state_file = tmp_path / "resumes" / ".mcp_watch_state.json"
    state_file.parent.mkdir()
    state_file.write_text("{}")
    monkeypatch.setenv("WATCH_STATE_FILE", str(state_file))

    ma._reset_watch_state(str(state_file.parent))
    assert not state_file.exists()
