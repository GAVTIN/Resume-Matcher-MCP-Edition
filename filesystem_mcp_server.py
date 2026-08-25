#!/usr/bin/env python3
"""
filesystem_mcp_server.py
=========================
MCP server that exposes the resume-matching pipeline's filesystem
operations as standardized, JSON-RPC 2.0 compliant MCP tools and
resources -- replacing the direct/custom filesystem tools used in
Milestone 1.

Run standalone (for the MCP Inspector, or manual poking):
    python filesystem_mcp_server.py

Run as a subprocess of an MCP client (this is how matching_agent.py
uses it, via langchain-mcp-adapters' MultiServerMCPClient over stdio):
    the client launches this exact file with `python filesystem_mcp_server.py`
    and speaks JSON-RPC 2.0 to it over stdin/stdout. Nothing in this
    file talks HTTP or knows about LangGraph -- that decoupling is the
    whole point of MCP.

Design notes
------------
* Transport: stdio. Simplest option, no network/auth surface to
  secure, and it's what langchain-mcp-adapters' MultiServerMCPClient
  expects for a local "command" server.
* Protocol compliance: the wire-level JSON-RPC 2.0 framing (requests,
  responses, the `initialize` handshake, `tools/list`, `tools/call`,
  `resources/list`, `resources/read`) is implemented by the official
  `mcp` SDK -- that's the point of building on it instead of hand-
  rolling JSON-RPC. This file's job is registering well-described
  tools/resources, validating inputs, and returning clean, structured
  success/error payloads on top of that.
* Error handling: every tool raises `mcp.server.fastmcp.exceptions
  .ToolError` on failure. FastMCP catches that and returns a
  CallToolResult with `isError=True`, which callers (including
  langchain-mcp-adapters) surface as a normal, catchable failure
  instead of crashing the transport. Each error message is a small
  JSON payload carrying a `code` in the JSON-RPC "server error" range
  (-32000 to -32099) plus a machine-readable `error` label, so callers
  can branch on it instead of string-matching prose. Verified against
  a live client session -- see tests/test_filesystem_mcp_server.py.
* Pinned to `mcp>=1.28,<2.0` (FastMCP, under `mcp.server.fastmcp`).
  The MCP Python SDK's v2 line, shipped for the 2026-07-28 spec
  revision, renames FastMCP to MCPServer under `mcp.server.mcpserver`.
  v1.x remains the actively supported, far more battle-tested line for
  the LangChain/LangGraph MCP ecosystem this project builds on, so
  this pin is deliberate, not stale -- see README "Design decisions".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("filesystem_mcp_server")


# --------------------------------------------------------------------------
# Configuration management
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    """Everything about this server's behaviour that should be tunable
    without touching code -- all overridable via environment variables.
    """

    resume_directory: Path
    results_directory: Path
    allowed_extensions: tuple[str, ...]
    max_batch_concurrency: int
    watch_state_file: Path

    @classmethod
    def from_env(cls) -> "ServerConfig":
        resume_dir = Path(os.getenv("RESUME_DIRECTORY", "./sample_data/resumes")).resolve()
        results_dir = Path(os.getenv("RESULTS_DIRECTORY", "./results")).resolve()
        raw_exts = os.getenv("ALLOWED_EXTENSIONS", ".txt,.pdf,.docx")
        exts = tuple(
            e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in raw_exts.split(",")
            if e.strip()
        )
        concurrency = int(os.getenv("MAX_BATCH_CONCURRENCY", "5"))
        watch_state = Path(
            os.getenv("WATCH_STATE_FILE", str(resume_dir / ".mcp_watch_state.json"))
        ).resolve()

        resume_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return cls(resume_dir, results_dir, exts, concurrency, watch_state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "resume_directory": str(self.resume_directory),
            "results_directory": str(self.results_directory),
            "allowed_extensions": list(self.allowed_extensions),
            "max_batch_concurrency": self.max_batch_concurrency,
            "watch_state_file": str(self.watch_state_file),
        }


CONFIG = ServerConfig.from_env()

mcp = FastMCP(
    "filesystem-resume-server",
    instructions=(
        "Filesystem tools for a resume-matching pipeline: list, read, "
        "search, and batch-extract resumes; watch a directory for newly "
        "added resumes; and persist match results."
    ),
)


# --------------------------------------------------------------------------
# Errors -- JSON-RPC reserves -32000..-32099 for implementation-defined
# "server error" codes; each domain error below gets its own code in
# that range so a caller can branch on `code`, not on message text.
# --------------------------------------------------------------------------

class ErrorCode:
    DIRECTORY_NOT_FOUND = -32001
    RESUME_NOT_FOUND = -32002
    UNSUPPORTED_FILE_TYPE = -32003
    EXTRACTION_FAILED = -32004
    INVALID_PARAMS = -32005


def _fail(code: int, error: str, message: str, **extra: Any) -> NoReturn:
    """Raise a ToolError carrying a small structured JSON payload.

    FastMCP converts any exception raised inside an @mcp.tool function
    into `CallToolResult(isError=True, content=[TextContent(...)])`.
    We put a JSON object -- not free prose -- in that text so a caller
    can `json.loads()` it and branch on `code`/`error`.
    """
    payload = {"success": False, "code": code, "error": error, "message": message, **extra}
    raise ToolError(json.dumps(payload))


# --------------------------------------------------------------------------
# Text extraction + path helpers (shared by several tools below)
# --------------------------------------------------------------------------

def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("pypdf is required to read .pdf resumes") from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx
        except ImportError as e:
            raise RuntimeError("python-docx is required to read .docx resumes") from e
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}")


def _resolve_directory(directory: str | None) -> Path:
    target = Path(directory).resolve() if directory else CONFIG.resume_directory
    if not target.exists() or not target.is_dir():
        _fail(
            ErrorCode.DIRECTORY_NOT_FOUND,
            "DIRECTORY_NOT_FOUND",
            f"Directory does not exist: {target}",
            directory=str(target),
        )
    return target


def _list_resume_paths(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in CONFIG.allowed_extensions and not p.name.startswith(".")
    )


def _resolve_resume_file(file_path: str) -> Path:
    """Accepts either a bare filename (looked up inside the configured
    resume directory) or a full/relative path.
    """
    candidate = Path(file_path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = CONFIG.resume_directory / file_path
    candidate = candidate.resolve()
    if not candidate.exists() or not candidate.is_file():
        _fail(
            ErrorCode.RESUME_NOT_FOUND,
            "RESUME_NOT_FOUND",
            f"No such resume file: {file_path}",
            file_path=file_path,
        )
    if candidate.suffix.lower() not in CONFIG.allowed_extensions:
        _fail(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "UNSUPPORTED_FILE_TYPE",
            f"{candidate.suffix} is not one of {list(CONFIG.allowed_extensions)}",
            file_path=file_path,
        )
    return candidate


# --------------------------------------------------------------------------
# Tools -- the Milestone 1 filesystem tools, now MCP tools
# --------------------------------------------------------------------------

@mcp.tool()
def list_resumes(directory: str | None = None) -> dict:
    """List resumes available for matching.

    Args:
        directory: Optional path to scan. Defaults to the server's
            configured RESUME_DIRECTORY.

    Returns the directory scanned and one entry per resume file
    (filename, path, size in bytes, last-modified timestamp).
    """
    target = _resolve_directory(directory)
    files = _list_resume_paths(target)
    return {
        "success": True,
        "directory": str(target),
        "count": len(files),
        "files": [
            {
                "filename": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime)),
            }
            for f in files
        ],
    }


@mcp.tool()
def read_resume(file_path: str) -> dict:
    """Read a single resume and extract its plain-text content.

    Args:
        file_path: Filename (resolved inside RESUME_DIRECTORY) or a
            full path. Supports .txt, .pdf, and .docx.
    """
    path = _resolve_resume_file(file_path)
    try:
        text = _extract_text(path)
    except Exception as e:
        _fail(
            ErrorCode.EXTRACTION_FAILED,
            "EXTRACTION_FAILED",
            f"Could not extract text from {path.name}: {e}",
            file_path=str(path),
        )
    return {
        "success": True,
        "file_path": str(path),
        "filename": path.name,
        "char_count": len(text),
        "text": text,
    }


@mcp.tool()
def search_resumes(query: str, directory: str | None = None) -> dict:
    """Keyword-search resume contents in a directory.

    Args:
        query: Case-insensitive keyword or short phrase to search for.
        directory: Optional path to scan. Defaults to RESUME_DIRECTORY.

    Returns matching files with a short snippet of surrounding context
    per match, so a caller can decide whether to pull the full text.
    """
    if not query or not query.strip():
        _fail(ErrorCode.INVALID_PARAMS, "INVALID_PARAMS", "query must be non-empty")

    target = _resolve_directory(directory)
    needle = query.strip().lower()
    matches = []
    for path in _list_resume_paths(target):
        try:
            text = _extract_text(path)
        except Exception as e:
            logger.warning("Skipping %s during search: %s", path.name, e)
            continue
        lower = text.lower()
        idx = lower.find(needle)
        if idx == -1:
            continue
        start, end = max(0, idx - 60), min(len(text), idx + len(needle) + 60)
        snippet = text[start:end].replace("\n", " ").strip()
        matches.append({"filename": path.name, "path": str(path), "snippet": f"...{snippet}..."})

    return {
        "success": True,
        "query": query,
        "directory": str(target),
        "match_count": len(matches),
        "matches": matches,
    }


@mcp.tool()
def save_match_result(
    candidate_file: str,
    job_id: str,
    score: float,
    summary: str,
    strengths: list[str] | None = None,
    gaps: list[str] | None = None,
) -> dict:
    """Persist a resume-to-job match result.

    Appends one JSON line to `<RESULTS_DIRECTORY>/match_results.jsonl`
    so results accumulate across runs and stay easy to tail/inspect.
    """
    record = {
        "candidate_file": candidate_file,
        "job_id": job_id,
        "score": score,
        "summary": summary,
        "strengths": strengths or [],
        "gaps": gaps or [],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = CONFIG.results_directory / "match_results.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"success": True, "saved_to": str(out_path), "record": record}


# --------------------------------------------------------------------------
# New MCP-specific capabilities
# --------------------------------------------------------------------------

@mcp.tool()
async def watch_directory(directory: str | None = None, ctx: Context | None = None) -> dict:
    """Report resumes added (or changed) since the last call.

    Poll-based watch: maintains a small JSON state file of previously
    seen filenames + modified-times next to the resume directory. The
    first call against a fresh directory reports every current file as
    "new" (a normal initial scan); later calls report only files that
    are new or whose mtime moved forward.

    This is deliberately polling rather than a filesystem-event
    listener (inotify/watchdog): MCP tools are request/response, so a
    caller (matching_agent.py) polls this on a timer, which is simpler
    to reason about and safer to demo than a background thread racing
    the event loop. See README "Design decisions" for the tradeoff and
    how a push-based version would work via MCP resource subscriptions.

    Args:
        directory: Optional path to watch. Defaults to RESUME_DIRECTORY.
    """
    target = _resolve_directory(directory)
    state_path = CONFIG.watch_state_file
    previous: dict[str, float] = {}
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            previous = {}

    current_files = _list_resume_paths(target)
    current_state = {f.name: f.stat().st_mtime for f in current_files}

    new_files = sorted(
        name for name, mtime in current_state.items()
        if name not in previous or mtime > previous[name] + 1e-6
    )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(current_state))

    if ctx is not None:
        await ctx.info(f"watch_directory: {len(new_files)} new file(s) in {target}")

    return {
        "success": True,
        "directory": str(target),
        "new_files": new_files,
        "new_count": len(new_files),
        "total_known": len(current_state),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@mcp.tool()
async def batch_process(
    directory: str | None = None,
    files: list[str] | None = None,
    operation: Literal["extract_text", "metadata_only"] = "extract_text",
    ctx: Context | None = None,
) -> dict:
    """Process multiple resumes concurrently instead of one call per file.

    Args:
        directory: Scan this directory for files to process. Ignored
            if `files` is given.
        files: Explicit filenames/paths to process -- e.g. only the
            new files `watch_directory` just reported. Takes priority
            over `directory`, so a caller can reprocess just what
            changed instead of the whole directory.
        operation: "extract_text" reads and extracts each file's text.
            "metadata_only" just stats the files (fast, no parsing).

    Runs with bounded concurrency (MAX_BATCH_CONCURRENCY, an
    asyncio.Semaphore) so a large inbox can't starve the event loop or
    blow through file-descriptor limits.
    """
    if files:
        targets = [_resolve_resume_file(f) for f in files]
    else:
        target_dir = _resolve_directory(directory)
        targets = _list_resume_paths(target_dir)

    if not targets:
        return {"success": True, "processed": 0, "succeeded": 0, "failed": 0, "results": [], "elapsed_seconds": 0.0}

    semaphore = asyncio.Semaphore(CONFIG.max_batch_concurrency)
    started = time.perf_counter()

    async def _process_one(path: Path) -> dict:
        async with semaphore:
            try:
                if operation == "metadata_only":
                    stat = path.stat()
                    return {"file": path.name, "path": str(path), "success": True, "size_bytes": stat.st_size}
                text = await asyncio.to_thread(_extract_text, path)
                return {"file": path.name, "path": str(path), "success": True, "char_count": len(text), "text": text}
            except Exception as e:
                return {"file": path.name, "path": str(path), "success": False, "error": str(e)}

    if ctx is not None:
        await ctx.info(f"batch_process: {len(targets)} file(s), concurrency={CONFIG.max_batch_concurrency}")

    results = list(await asyncio.gather(*(_process_one(p) for p in targets)))
    succeeded = sum(1 for r in results if r["success"])

    return {
        "success": True,
        "processed": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


# --------------------------------------------------------------------------
# Resources -- discoverable, read-only, no side effects (unlike tools)
# --------------------------------------------------------------------------

@mcp.resource("config://server-info", mime_type="application/json")
def server_info() -> str:
    """Current server configuration and capabilities (resource discovery)."""
    return json.dumps(
        {
            "name": "filesystem-resume-server",
            "capabilities": [
                "list_resumes", "read_resume", "search_resumes",
                "save_match_result", "watch_directory", "batch_process",
            ],
            "config": CONFIG.as_dict(),
        },
        indent=2,
    )


@mcp.resource("resume://inbox", mime_type="application/json")
def resume_inbox() -> str:
    """Read-only listing of the configured resume directory."""
    return json.dumps([f.name for f in _list_resume_paths(CONFIG.resume_directory)], indent=2)


@mcp.resource("resume://{filename}", mime_type="text/plain")
def resume_content(filename: str) -> str:
    """Raw extracted text of one resume, addressed by filename."""
    return _extract_text(_resolve_resume_file(filename))


if __name__ == "__main__":
    logger.info("Starting filesystem MCP server | config=%s", CONFIG.as_dict())
    mcp.run(transport="stdio")
