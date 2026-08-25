"""
tests/test_filesystem_mcp_server.py
====================================
Two layers of testing:

1. Unit tests that call the tool functions directly (no subprocess,
   no protocol) -- fast feedback on business logic.
2. An integration test that launches filesystem_mcp_server.py as a
   real subprocess and drives it with the official MCP client SDK
   over stdio -- proves the JSON-RPC 2.0 protocol layer actually
   works, not just the underlying Python.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def resume_env(tmp_path, monkeypatch):
    """Isolated resume/results dirs, with a fresh import of the server
    module so its module-level CONFIG (built once, at import time)
    picks up these env vars instead of the real sample_data/ directory.
    """
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "alice.txt").write_text("Alice Nguyen\nPython, PostgreSQL, AWS, 5 years backend.")
    (resumes / "bob.txt").write_text("Bob Lee\nReact, TypeScript, some Node.js.")
    results = tmp_path / "results"

    monkeypatch.setenv("RESUME_DIRECTORY", str(resumes))
    monkeypatch.setenv("RESULTS_DIRECTORY", str(results))
    monkeypatch.setenv("WATCH_STATE_FILE", str(resumes / ".mcp_watch_state.json"))

    sys.modules.pop("filesystem_mcp_server", None)
    import filesystem_mcp_server as server  # imported after env vars are set

    return server, resumes, results


# ---------------------------------------------------------------------
# Unit tests -- business logic only
# ---------------------------------------------------------------------

def test_list_resumes_finds_only_allowed_extensions(resume_env):
    server, resumes, _ = resume_env
    (resumes / "notes.md").write_text("not a resume, should be ignored")
    result = server.list_resumes()
    assert result["success"] is True
    assert result["count"] == 2
    assert {f["filename"] for f in result["files"]} == {"alice.txt", "bob.txt"}


def test_read_resume_extracts_text(resume_env):
    server, _, _ = resume_env
    result = server.read_resume(file_path="alice.txt")
    assert result["success"] is True
    assert "PostgreSQL" in result["text"]


def test_read_resume_missing_file_raises_structured_error(resume_env):
    server, _, _ = resume_env
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        server.read_resume(file_path="does_not_exist.txt")
    payload = json.loads(str(exc_info.value))
    assert payload["success"] is False
    assert payload["code"] == server.ErrorCode.RESUME_NOT_FOUND
    assert payload["error"] == "RESUME_NOT_FOUND"


def test_search_resumes_finds_keyword_with_snippet(resume_env):
    server, _, _ = resume_env
    result = server.search_resumes(query="postgresql")
    assert result["match_count"] == 1
    assert result["matches"][0]["filename"] == "alice.txt"
    assert "postgresql" in result["matches"][0]["snippet"].lower()


def test_search_resumes_rejects_empty_query(resume_env):
    server, _, _ = resume_env
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        server.search_resumes(query="   ")
    payload = json.loads(str(exc_info.value))
    assert payload["code"] == server.ErrorCode.INVALID_PARAMS


def test_save_match_result_appends_jsonl(resume_env):
    server, _, results = resume_env
    out = server.save_match_result(
        candidate_file="alice.txt", job_id="JOB-1", score=88.5,
        summary="Strong fit", strengths=["AWS"], gaps=[],
    )
    assert out["success"] is True
    lines = (results / "match_results.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["candidate_file"] == "alice.txt"


@pytest.mark.asyncio
async def test_watch_directory_reports_new_then_settles_then_detects_addition(resume_env):
    server, resumes, _ = resume_env

    first = await server.watch_directory()
    assert first["new_count"] == 2  # alice + bob: first-ever scan reports everything

    second = await server.watch_directory()
    assert second["new_count"] == 0  # nothing changed since the last check

    (resumes / "carol.txt").write_text("Carol Diaz\nGo, Kubernetes, distributed systems.")
    third = await server.watch_directory()
    assert third["new_files"] == ["carol.txt"]


@pytest.mark.asyncio
async def test_batch_process_runs_concurrently_and_reports_partial_failure(resume_env):
    server, resumes, _ = resume_env
    (resumes / "corrupt.pdf").write_bytes(b"this is not a real pdf")

    result = await server.batch_process(directory=str(resumes))
    assert result["processed"] == 3
    assert result["succeeded"] == 2
    assert result["failed"] == 1
    failed_files = {r["file"] for r in result["results"] if not r["success"]}
    assert failed_files == {"corrupt.pdf"}


@pytest.mark.asyncio
async def test_batch_process_honors_explicit_file_list(resume_env):
    """This is the path matching_agent.py actually uses: reprocess only
    the specific files watch_directory just reported, not the whole dir.
    """
    server, _, _ = resume_env
    result = await server.batch_process(files=["alice.txt"])
    assert result["processed"] == 1
    assert result["results"][0]["file"] == "alice.txt"


# ---------------------------------------------------------------------
# Integration test -- real JSON-RPC 2.0 over stdio, via the MCP client SDK
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_server_speaks_mcp_protocol_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "dana.txt").write_text("Dana Kim\nDjango, PostgreSQL, Docker, AWS.")

    env = {
        **os.environ,
        "RESUME_DIRECTORY": str(resumes),
        "RESULTS_DIRECTORY": str(tmp_path / "results"),
        "WATCH_STATE_FILE": str(resumes / ".mcp_watch_state.json"),
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "filesystem_mcp_server.py")],
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Resource discovery
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert {"list_resumes", "read_resume", "watch_directory", "batch_process"} <= tool_names

            resources = await session.list_resources()
            resource_uris = {str(r.uri) for r in resources.resources}
            assert "config://server-info" in resource_uris or "resume://inbox" in resource_uris

            # Successful tool call
            ok = await session.call_tool("list_resumes", {})
            assert ok.isError is False
            payload = json.loads(ok.content[0].text)
            assert payload["count"] == 1

            # Failed tool call -> isError=True with our structured JSON payload
            bad = await session.call_tool("read_resume", {"file_path": "ghost.txt"})
            assert bad.isError is True
            assert "RESUME_NOT_FOUND" in bad.content[0].text

            # Resource read
            config_resource = await session.read_resource("config://server-info")
            config_payload = json.loads(config_resource.contents[0].text)
            assert config_payload["name"] == "filesystem-resume-server"
