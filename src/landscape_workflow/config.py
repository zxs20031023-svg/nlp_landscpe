from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_TITLE = "景观设计需求转译与场地分析工作流"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_MODELS = [
    "qwen-plus",
    "qwen-max",
    "qwen-turbo",
    "deepseek-chat",
    "glm-4",
    "gpt-4o",
    "自定义输入...",
]


@dataclass(frozen=True)
class AppPaths:
    root: Path
    src_dir: Path
    package_dir: Path
    docs_dir: Path
    config_dir: Path
    resources_dir: Path
    runtime_dir: Path
    temp_dir: Path
    chroma_db: Path
    output_dir: Path
    history_dir: Path
    digital_briefs_dir: Path
    sense_mapping_file: Path
    compliance_rules_file: Path
    knowledge_aliases_file: Path
    project_case_library_file: Path
    knowledge_base_dir: Path
    sample_dir: Path


ROOT = Path(__file__).resolve().parents[2]
PATHS = AppPaths(
    root=ROOT,
    src_dir=ROOT / "src",
    package_dir=ROOT / "src" / "landscape_workflow",
    docs_dir=ROOT / "docs",
    config_dir=ROOT / "config",
    resources_dir=ROOT / "resources",
    runtime_dir=ROOT / "runtime",
    temp_dir=ROOT / "runtime" / "temp",
    chroma_db=ROOT / "runtime" / "chroma_db",
    output_dir=ROOT / "runtime" / "output_briefs",
    history_dir=ROOT / "runtime" / "output_briefs",
    digital_briefs_dir=ROOT / "runtime" / "output_briefs" / "digital_briefs",
    sense_mapping_file=ROOT / "config" / "sense_mapping.json",
    compliance_rules_file=ROOT / "config" / "compliance_rules.json",
    knowledge_aliases_file=ROOT / "config" / "knowledge_aliases.json",
    project_case_library_file=ROOT / "resources" / "reference_projects" / "project_case_library.json",
    knowledge_base_dir=ROOT / "resources" / "knowledge_base",
    sample_dir=ROOT / "resources" / "samples",
)


def ensure_runtime_dirs() -> None:
    PATHS.runtime_dir.mkdir(parents=True, exist_ok=True)
    PATHS.temp_dir.mkdir(parents=True, exist_ok=True)
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    PATHS.history_dir.mkdir(parents=True, exist_ok=True)
    PATHS.digital_briefs_dir.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_recent_history(limit: int = 10) -> list[dict]:
    ensure_runtime_dirs()
    files = sorted(PATHS.history_dir.glob("history_*.json"), reverse=True)[:limit]
    records: list[dict] = []
    for file_path in files:
        try:
            records.append(json.loads(file_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records
