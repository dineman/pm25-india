"""Archive a Claude Code session transcript and write a readable Markdown copy.

Run automatically by the SessionEnd hook in .claude/settings.json, which pipes
the hook payload (JSON with ``transcript_path`` and ``session_id``) to stdin.
It can also be run by hand on any saved transcript:

    python3 scripts/export_session.py --transcript ~/.claude/projects/<proj>/<id>.jsonl

Writes logs/sessions/<YYYY-MM-DD>_<id8>.jsonl (raw copy) and .md next to it.
Uses only the standard library so the hook works without activating conda.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

MAX_RESULT_CHARS = 2000
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "logs" / "sessions"


def read_entries(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL transcript, skipping lines that are not valid JSON."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Shorten long text, noting how much was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters]"


def result_text(content: Any) -> str:
    """Flatten a tool_result content field (string or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        images = sum(1 for b in content if b.get("type") == "image")
        if images:
            parts.append(f"[{images} image(s)]")
        return "\n".join(parts)
    return ""


def render_block(role: str, block: dict[str, Any]) -> str | None:
    """Render one content block as Markdown, or None to skip it."""
    kind = block.get("type")
    if kind == "text" and block.get("text", "").strip():
        heading = "## You" if role == "user" else "## Claude"
        return f"{heading}\n\n{block['text'].strip()}"
    if kind == "tool_use":
        args = json.dumps(block.get("input", {}), indent=2, ensure_ascii=False)
        return f"**Tool: {block.get('name')}**\n\n```json\n{truncate(args)}\n```"
    if kind == "tool_result":
        text = truncate(result_text(block.get("content")).strip())
        return f"<details><summary>Result</summary>\n\n```\n{text}\n```\n\n</details>"
    return None


def to_markdown(entries: list[dict[str, Any]], session_id: str) -> str:
    """Convert transcript entries to Markdown: prompts, replies, tool calls."""
    stamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    lines = [f"# Claude Code session {session_id}", ""]
    if stamps:
        lines += [f"{stamps[0]} to {stamps[-1]}", ""]

    for entry in entries:
        role = entry.get("type")
        if role not in ("user", "assistant") or entry.get("isMeta"):
            continue
        if entry.get("isSidechain"):
            continue
        content = entry.get("message", {}).get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content or []:
            rendered = render_block(role, block)
            if rendered:
                lines += [rendered, ""]
    return "\n".join(lines)


def export(transcript: Path, session_id: str, out_dir: Path) -> tuple[Path, Path]:
    """Copy the transcript and write its Markdown rendering; return both paths."""
    entries = read_entries(transcript)
    stamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    date = stamps[0][:10] if stamps else "unknown-date"
    stem = f"{date}_{session_id[:8]}"

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{stem}.jsonl"
    md = out_dir / f"{stem}.md"
    shutil.copyfile(transcript, raw)
    md.write_text(to_markdown(entries, session_id), encoding="utf-8")
    return raw, md


def parse_args() -> argparse.Namespace:
    """Parse arguments; without --transcript the hook payload is read from stdin."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript", type=Path, help="transcript .jsonl to export")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Export the transcript named on the command line or in the hook payload."""
    args = parse_args()
    if args.transcript:
        transcript = args.transcript.expanduser()
        session_id = transcript.stem
    else:
        payload = json.load(sys.stdin)
        transcript = Path(payload["transcript_path"]).expanduser()
        session_id = payload.get("session_id") or transcript.stem

    if not transcript.exists():
        sys.exit(f"Transcript not found: {transcript}")
    raw, md = export(transcript, session_id, args.out_dir)
    print(f"Saved {raw} and {md}")


if __name__ == "__main__":
    main()
