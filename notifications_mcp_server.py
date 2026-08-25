#!/usr/bin/env python3
"""
notifications_mcp_server.py
============================
A second, independent MCP server -- used to demonstrate Part B's
"Multi-MCP Integration" bonus: matching_agent.py talks to this server
*and* filesystem_mcp_server.py at the same time, through the same
MultiServerMCPClient, with no knowledge of each other.

In a real deployment this would call an email/Slack API. Here it
writes to a local, human-readable log instead, so the demo doesn't
need real credentials and stays deterministic and offline-safe. The
MCP-integration pattern (a second stdio server, aggregated by
MultiServerMCPClient alongside the filesystem one) is the point being
demonstrated, not the notification transport itself -- swapping the
body of `send_match_notification` for a real Slack/SMTP call would be
a small, contained change.

Run standalone: python notifications_mcp_server.py
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("notifications_mcp_server")


@dataclass(frozen=True)
class NotifyConfig:
    log_path: Path
    default_recipient: str

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        log_path = Path(os.getenv("NOTIFICATIONS_LOG", "./results/notifications.log")).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        recipient = os.getenv("DEFAULT_NOTIFY_RECIPIENT", "hiring-team@example.com")
        return cls(log_path, recipient)


CONFIG = NotifyConfig.from_env()

mcp = FastMCP(
    "notifications-server",
    instructions="Send and list match notifications. Standalone from the filesystem server on purpose.",
)


@mcp.tool()
def send_match_notification(
    candidate_file: str,
    job_id: str,
    score: float,
    recipient: str | None = None,
) -> dict:
    """Notify a recipient about a strong resume match.

    Args:
        candidate_file: The resume filename this notification is about.
        job_id: The job description identifier that was matched against.
        score: Match score (0-100) that triggered this notification.
        recipient: Who to notify. Defaults to DEFAULT_NOTIFY_RECIPIENT.

    Writes a formatted, timestamped line to the notifications log
    (simulating a send) and returns the record so a caller can verify
    exactly what would have gone out.
    """
    if not (0 <= score <= 100):
        raise ToolError(json.dumps({
            "success": False, "code": -32005, "error": "INVALID_PARAMS",
            "message": "score must be between 0 and 100",
        }))

    record = {
        "candidate_file": candidate_file,
        "job_id": job_id,
        "score": score,
        "recipient": recipient or CONFIG.default_recipient,
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    line = (
        f"[{record['sent_at']}] To: {record['recipient']} | "
        f"Candidate: {record['candidate_file']} | Job: {record['job_id']} | Score: {record['score']}"
    )
    with CONFIG.log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    return {"success": True, "logged_to": str(CONFIG.log_path), "record": record}


@mcp.tool()
def list_notifications(limit: int = 10) -> dict:
    """Return the most recent notifications sent.

    Args:
        limit: Max number of most-recent entries to return (default 10).
    """
    if not CONFIG.log_path.exists():
        return {"success": True, "count": 0, "notifications": []}
    lines = CONFIG.log_path.read_text(encoding="utf-8").splitlines()
    recent = lines[-limit:] if limit > 0 else lines
    return {"success": True, "count": len(recent), "notifications": recent}


@mcp.resource("notifications://recent", mime_type="text/plain")
def recent_notifications() -> str:
    """Read-only view of the notification log's last 20 lines."""
    if not CONFIG.log_path.exists():
        return "(no notifications sent yet)"
    lines = CONFIG.log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-20:]) or "(no notifications sent yet)"


if __name__ == "__main__":
    logger.info("Starting notifications MCP server | log=%s", CONFIG.log_path)
    mcp.run(transport="stdio")
