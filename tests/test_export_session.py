import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_session.py"


def write_transcript(path: Path) -> None:
    entries = [
        {"type": "mode", "mode": "auto"},
        {
            "type": "user",
            "timestamp": "2026-10-18T06:00:00Z",
            "message": {"role": "user", "content": "how do i log an error"},
        },
        {
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-10-18T06:00:01Z",
            "message": {"role": "user", "content": "hidden system note"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-18T06:00:05Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "Use logging."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-10-18T06:00:06Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "x" * 5000}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\nnot json\n")


def test_hook_payload_exports_raw_and_markdown(tmp_path: Path) -> None:
    transcript = tmp_path / "abcdef1234.jsonl"
    write_transcript(transcript)
    out_dir = tmp_path / "out"
    payload = json.dumps(
        {"transcript_path": str(transcript), "session_id": "abcdef1234"}
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(out_dir)],
        input=payload,
        text=True,
        check=True,
        capture_output=True,
    )

    raw = out_dir / "2026-10-18_abcdef12.jsonl"
    md = (out_dir / "2026-10-18_abcdef12.md").read_text()
    assert raw.read_bytes() == transcript.read_bytes()
    assert "## You\n\nhow do i log an error" in md
    assert "## Claude\n\nUse logging." in md
    assert "**Tool: Bash**" in md
    assert "more characters" in md
    assert "hidden system note" not in md
    assert "private" not in md
