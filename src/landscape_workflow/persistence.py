from __future__ import annotations

import json
import re
from pathlib import Path

from .config import PATHS, ensure_runtime_dirs
from .models import HistoryRecord


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "default_project"


def _write_json(file_path: Path, payload: dict) -> Path:
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path


def save_history_record(record: HistoryRecord) -> Path:
    ensure_runtime_dirs()
    file_path = PATHS.history_dir / f"history_{record.created_at.replace(':', '-')}_{record.run_id}.json"
    return _write_json(file_path, record.model_dump(mode="json"))


def save_digital_brief(
    *,
    project_name: str,
    created_at: str,
    run_id: str,
    payload: dict,
) -> Path:
    ensure_runtime_dirs()
    safe_project_name = _sanitize_filename(project_name)
    timestamp = created_at.replace(":", "-")
    file_path = PATHS.digital_briefs_dir / f"{safe_project_name}_digital_brief_{timestamp}_{run_id}.json"
    return _write_json(file_path, payload)
