#!/usr/bin/env python3
"""
matching_agent.py
==================
Resume-matching agent, refactored from Milestone 1's direct/custom
filesystem tools onto MCP. Instead of importing filesystem functions
directly, the agent connects to filesystem_mcp_server.py (and, for the
multi-MCP bonus, notifications_mcp_server.py) as a client, over stdio,
via langchain-mcp-adapters' MultiServerMCPClient -- and discovers its
tools at startup rather than hard-coding them.

Usage:
    export ANTHROPIC_API_KEY=<your-api-key>
    python matching_agent.py
    python matching_agent.py --job-description sample_data/job_description.txt \\
                              --resume-dir sample_data/resumes
        python matching_agent.py --watch --interval 15   # keep polling for new resumes
        python matching_agent.py --reset-watch            # explicitly forget watcher state

State machine
-------------
    START -> check_new_resumes -> [has_new?] --no--> END
                                        |
                                       yes
                                        v
                                  batch_extract -> match -> rank_and_save -> notify -> END

See README.md for the rendered diagram and the reasoning behind each
node. The short version: `check_new_resumes` calls the filesystem
server's `watch_directory` tool, so a run with nothing new to do
short-circuits without spending an LLM call -- and `notify` is the
only node that talks to the *second* MCP server, which is what
demonstrates multi-MCP integration end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Literal, TypedDict

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("matching_agent")

HERE = Path(__file__).parent.resolve()
NOTIFY_THRESHOLD = float(os.getenv("NOTIFY_SCORE_THRESHOLD", "70"))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "3"))


def _require_model_credentials(model_name: str) -> None:
    """Fail before MCP work when the selected provider cannot authenticate."""
    provider = model_name.split(":", 1)[0].lower()
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in this terminal. "
            "In PowerShell run: $env:ANTHROPIC_API_KEY = '<your-api-key>'"
        )

def _build_mcp_servers() -> dict[str, dict[str, Any]]:
    """MCP servers this agent connects to. Built fresh on each call
    (rather than as a module-level constant) so it always reflects the
    *current* environment -- see the `env` note below.

    Both servers are launched as local subprocesses over stdio -- add
    more entries here (or point a "transport": "streamable-http" entry
    at a remote MCP server) and every node below keeps working
    unchanged, because nodes look tools up by name from the aggregated
    pool rather than importing them.

    NOTE on `env`: the MCP stdio client does NOT inherit the parent
    process's environment by default -- it starts child servers with a
    minimal, curated env (PATH/HOME/TERM only), as a deliberate
    security default so a server can't silently harvest whatever's in
    the caller's environment. Confirmed directly against `mcp.client
    .stdio.get_default_environment()` while building this. That means
    without passing `env` explicitly, RESUME_DIRECTORY /
    RESULTS_DIRECTORY / etc. would silently fail to reach
    filesystem_mcp_server.py. Forwarding the full parent environment is
    the right call here since both servers are ours, running locally,
    in the same project -- a third-party server would warrant a
    narrower allow-list instead.
    """
    env = dict(os.environ)
    return {
        "filesystem": {
            "command": sys.executable,
            "args": [str(HERE / "filesystem_mcp_server.py")],
            "transport": "stdio",
            "env": env,
        },
        "notifications": {
            "command": sys.executable,
            "args": [str(HERE / "notifications_mcp_server.py")],
            "transport": "stdio",
            "env": env,
        },
    }


# Kept for readability / anyone importing this to inspect server
# names+commands without an env snapshot; MatchingAgent itself calls
# _build_mcp_servers() fresh in __init__, not this.
MCP_SERVERS: dict[str, dict[str, Any]] = _build_mcp_servers()


# --------------------------------------------------------------------------
# Structured LLM output for the "match" node
# --------------------------------------------------------------------------

class MatchAssessment(BaseModel):
    """What the LLM produces for one resume against the job description."""

    score: float = Field(ge=0, le=100, description="Overall fit score, 0-100")
    summary: str = Field(description="1-3 sentence overall assessment")
    strengths: list[str] = Field(default_factory=list, description="Concrete strengths relevant to the JD")
    gaps: list[str] = Field(default_factory=list, description="Concrete gaps or missing requirements")

    @field_validator("strengths", "gaps", mode="before")
    @classmethod
    def _normalize_list_fields(cls, value: Any) -> list[str]:
        """Anthropic sometimes returns a single string instead of a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []

            xml_items = re.findall(r"<item>(.*?)</item>", text, flags=re.DOTALL | re.IGNORECASE)
            if xml_items:
                return [item.strip() for item in xml_items if item.strip()]

            parts = re.split(r"[\n;]+", text)
            cleaned = [p.strip().lstrip("-*• ").strip() for p in parts if p.strip()]
            return cleaned or [text]
        return [str(value)]


def _coerce_assessment(raw: Any) -> MatchAssessment:
    if isinstance(raw, MatchAssessment):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = raw
    else:
        data = {"score": 0, "summary": str(raw), "strengths": [], "gaps": []}

    normalized = {
        "score": float(data.get("score", 0)),
        "summary": str(data.get("summary", "")),
        "strengths": data.get("strengths", []),
        "gaps": data.get("gaps", []),
    }
    return MatchAssessment(**normalized)


# --------------------------------------------------------------------------
# Agent state
# --------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    job_description: str
    job_id: str
    resume_directory: str
    new_files: list[str]
    extracted: list[dict]       # [{file, path, success, text/error}, ...]
    matches: list[dict]         # [{candidate_file, job_id, score, summary, strengths, gaps}, ...]
    notified: list[str]
    log: list[str]


def _log(state: AgentState, message: str) -> list[str]:
    logger.info(message)
    return [*state.get("log", []), message]


# --------------------------------------------------------------------------
# MCP call helper
# --------------------------------------------------------------------------

class MCPToolCallError(RuntimeError):
    """Raised when an MCP tool call comes back with success=False.

    Carries the structured error payload the server sent (see
    filesystem_mcp_server.py's `_fail`) so callers can branch on
    `.payload["code"]` instead of parsing prose.
    """

    def __init__(self, tool_name: str, payload: dict[str, Any]):
        self.tool_name = tool_name
        self.payload = payload
        super().__init__(
            f"MCP tool '{tool_name}' failed "
            f"[{payload.get('code')}] {payload.get('error')}: {payload.get('message')}"
        )


def _extract_text(raw_result: Any) -> str:
    """langchain-mcp-adapters returns tool output as a list of content
    blocks (`[{"type": "text", "text": "..."}]`), not a plain string.
    """
    if isinstance(raw_result, str):
        return raw_result
    if isinstance(raw_result, list) and raw_result:
        first = raw_result[0]
        if isinstance(first, dict):
            return first.get("text", str(first))
        return getattr(first, "text", str(first))
    return str(raw_result)


def _parse_tool_result(text: str) -> dict[str, Any]:
    """Parse a tool's text content into a dict.

    A failed tool call comes back as `'Error executing tool X: {json}'`
    (FastMCP's ToolError wrapping, observed directly against a live
    server in tests/test_filesystem_mcp_server.py) -- so we locate the
    embedded JSON object rather than assuming the whole string parses.
    """
    marker = text.find("{")
    if text.startswith("Error executing tool") and marker != -1:
        try:
            return json.loads(text[marker:])
        except json.JSONDecodeError:
            return {"success": False, "code": None, "error": "UNKNOWN", "message": text}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"success": True, "raw": text}


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class MatchingAgent:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("MATCHING_AGENT_MODEL", "anthropic:claude-sonnet-5")
        _require_model_credentials(self.model_name)
        self.client = MultiServerMCPClient(_build_mcp_servers())
        if self.model_name.startswith("anthropic:"):
            # Anthropic's newer models reject the `temperature` parameter.
            self.model = init_chat_model(self.model_name)
        else:
            self.model = init_chat_model(self.model_name, temperature=0)
        self._tool_map: dict[str, Any] = {}
        self.graph = None

    async def setup(self) -> None:
        """Discover tools from every configured MCP server and compile the graph.

        This is the crux of the Milestone 1 -> MCP refactor: the agent
        no longer imports filesystem functions directly. It asks the
        MCP servers what they expose and looks tools up by name, so
        adding a capability to filesystem_mcp_server.py doesn't require
        touching this file at all.
        """
        tools = await self.client.get_tools()
        self._tool_map = {t.name: t for t in tools}
        logger.info("Discovered %d MCP tool(s): %s", len(tools), sorted(self._tool_map))
        self.graph = self._build_graph()

    async def _call_tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        tool = self._tool_map.get(name)
        if tool is None:
            raise RuntimeError(
                f"MCP tool '{name}' was not discovered from any connected server "
                f"(have: {sorted(self._tool_map)}). Did setup() run?"
            )
        raw = await tool.ainvoke(kwargs)
        parsed = _parse_tool_result(_extract_text(raw))
        if not parsed.get("success", True):
            raise MCPToolCallError(name, parsed)
        return parsed

    # ---- graph nodes -----------------------------------------------------

    async def node_check_new_resumes(self, state: AgentState) -> dict[str, Any]:
        result = await self._call_tool("watch_directory", directory=state.get("resume_directory"))
        log = _log(state, f"check_new_resumes: {result['new_count']} new file(s) -> {result['new_files']}")
        return {"new_files": result["new_files"], "log": log}

    async def node_batch_extract(self, state: AgentState) -> dict[str, Any]:
        result = await self._call_tool("batch_process", files=state["new_files"], operation="extract_text")
        log = _log(
            state,
            f"batch_extract: {result['succeeded']}/{result['processed']} succeeded "
            f"in {result['elapsed_seconds']}s",
        )
        return {"extracted": result["results"], "log": log}

    async def node_match(self, state: AgentState) -> dict[str, Any]:
        structured_model = self.model.with_structured_output(MatchAssessment)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

        async def _score_one(item: dict) -> dict | None:
            if not item.get("success"):
                logger.warning("Skipping %s (extraction failed: %s)", item.get("file"), item.get("error"))
                return None
            prompt = (
                "You are screening a candidate resume against a job description. "
                "Score fit from 0-100 and list concrete strengths and gaps.\n\n"
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"RESUME ({item['file']}):\n{item['text']}"
            )
            async with semaphore:
                raw_assessment = await structured_model.ainvoke(prompt)
            assessment = _coerce_assessment(raw_assessment)
            return {
                "candidate_file": item["file"],
                "job_id": state["job_id"],
                **assessment.model_dump(),
            }

        scored = await asyncio.gather(*(_score_one(item) for item in state["extracted"]))
        matches = [m for m in scored if m is not None]
        matches.sort(key=lambda m: m["score"], reverse=True)
        log = _log(state, f"match: scored {len(matches)} resume(s)")
        return {"matches": matches, "log": log}

    async def node_rank_and_save(self, state: AgentState) -> dict[str, Any]:
        for m in state["matches"]:
            await self._call_tool(
                "save_match_result",
                candidate_file=m["candidate_file"],
                job_id=m["job_id"],
                score=m["score"],
                summary=m["summary"],
                strengths=m["strengths"],
                gaps=m["gaps"],
            )
        log = _log(state, f"rank_and_save: persisted {len(state['matches'])} result(s)")
        return {"log": log}

    async def node_notify(self, state: AgentState) -> dict[str, Any]:
        notified = []
        for m in state["matches"]:
            if m["score"] >= NOTIFY_THRESHOLD:
                await self._call_tool(
                    "send_match_notification",
                    candidate_file=m["candidate_file"],
                    job_id=m["job_id"],
                    score=m["score"],
                )
                notified.append(m["candidate_file"])
        log = _log(
            state,
            f"notify: sent {len(notified)} notification(s) "
            f"(threshold={NOTIFY_THRESHOLD}) via the notifications MCP server",
        )
        return {"notified": notified, "log": log}

    # ---- graph wiring ------------------------------------------------------

    def _route_after_check(self, state: AgentState) -> Literal["batch_extract", "__end__"]:
        return "batch_extract" if state.get("new_files") else END

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("check_new_resumes", self.node_check_new_resumes)
        builder.add_node("batch_extract", self.node_batch_extract)
        builder.add_node("match", self.node_match)
        builder.add_node("rank_and_save", self.node_rank_and_save)
        builder.add_node("notify", self.node_notify)

        builder.add_edge(START, "check_new_resumes")
        builder.add_conditional_edges(
            "check_new_resumes", self._route_after_check, {"batch_extract": "batch_extract", END: END}
        )
        builder.add_edge("batch_extract", "match")
        builder.add_edge("match", "rank_and_save")
        builder.add_edge("rank_and_save", "notify")
        builder.add_edge("notify", END)
        return builder.compile()

    # ---- public entrypoint ------------------------------------------------

    async def run(self, job_description: str, job_id: str, resume_directory: str) -> AgentState:
        if self.graph is None:
            await self.setup()
        initial: AgentState = {
            "job_description": job_description,
            "job_id": job_id,
            "resume_directory": resume_directory,
            "log": [],
        }
        return await self.graph.ainvoke(initial)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_summary(final_state: AgentState) -> None:
    matches = final_state.get("matches")
    if not matches:
        print("\nNo new resumes to score this run.")
        return
    print(f"\n{'Candidate':35s} {'Score':>6s}  Summary")
    print("-" * 90)
    for m in matches:
        print(f"{m['candidate_file']:35s} {m['score']:6.1f}  {m['summary'][:70]}")
    notified = final_state.get("notified", [])
    if notified:
        print(f"\nNotified hiring team about {len(notified)} strong match(es): {', '.join(notified)}")


def _reset_watch_state(resume_directory: str) -> None:
    watch_state = Path(
        os.getenv("WATCH_STATE_FILE", str(Path(resume_directory) / ".mcp_watch_state.json"))
    )
    watch_state.unlink(missing_ok=True)


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Resume-matching agent (MCP edition)")
    parser.add_argument("--job-description", default=str(HERE / "sample_data" / "job_description.txt"))
    parser.add_argument("--job-id", default=None, help="Defaults to the job description filename stem")
    parser.add_argument("--resume-dir", default=str(HERE / "sample_data" / "resumes"))
    parser.add_argument("--watch", action="store_true", help="Keep polling resume-dir for new files")
    parser.add_argument("--interval", type=float, default=15.0, help="Seconds between polls in --watch mode")
    parser.add_argument(
        "--reset-watch",
        action="store_true",
        help="Forget previously seen resumes before the first scan",
    )
    args = parser.parse_args()

    jd_path = Path(args.job_description)
    job_description = jd_path.read_text(encoding="utf-8")
    job_id = args.job_id or jd_path.stem

    if args.reset_watch or not args.watch:
        _reset_watch_state(args.resume_dir)

    agent = MatchingAgent()
    await agent.setup()

    while True:
        final_state = await agent.run(job_description, job_id, args.resume_dir)
        _print_summary(final_state)
        if not args.watch:
            break
        print(f"\n--watch mode: sleeping {args.interval}s. Drop a new resume into {args.resume_dir} to see it picked up. Ctrl+C to stop.")
        await asyncio.sleep(args.interval)


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
