from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class WorkflowSettings(BaseModel):
    project_name: str = "默认项目"
    api_key: str
    base_url: str
    model_name: str
    embedding_model: str
    retrieval_top_k: int = 4


class LandscapeBrief(BaseModel):
    project_type: str = Field(description="项目类型，如社区口袋公园、社区游园")
    style_preference: str = Field(description="风格偏好，如生态适老、现代自然")
    canopy_closure: float = Field(description="植物郁闭度，取值区间 0.0-1.0")
    path_slope_max_percentage: float = Field(description="园路最大坡度，单位为百分比")
    hardscape_ratio: float = Field(description="硬质铺装比例，取值区间 0.0-1.0")
    functional_zones: list[str] = Field(description="功能分区列表")
    warnings: list[str] = Field(default_factory=list, description="风险与修正说明")


class SiteAnalysis(BaseModel):
    location_context: str = Field(description="区位与周边环境分析")
    climate_environment: str = Field(description="气候与微环境分析")
    topography_features: str = Field(description="地形地貌与水文分析")
    opportunities: list[str] = Field(description="机会点")
    constraints: list[str] = Field(description="约束条件")
    design_suggestions: list[str] = Field(description="设计建议")


class RetrievalHit(BaseModel):
    source: str
    excerpt: str


class AppliedRule(BaseModel):
    rule_id: str
    title: str
    warning: str
    payload: dict = Field(default_factory=dict)


class ReferenceProject(BaseModel):
    project_id: str
    name: str
    city: str
    project_type: str
    scene: str
    summary: str
    source_url: str = ""
    source_label: str = "本地案例库"
    library_source: str = "local"
    keywords: list[str] = Field(default_factory=list)
    target_users: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    matching_points: list[str] = Field(default_factory=list)
    recommendation_reason: str = ""
    similarity_score: float = 0.0


class KnowledgeFileRecord(BaseModel):
    filename: str
    file_hash: str
    chunk_count: int
    ingested_at: str
    source_path: str


class KnowledgeBaseStats(BaseModel):
    ready: bool
    document_count: int = 0
    documents: list[KnowledgeFileRecord] = Field(default_factory=list)


class IngestionResult(BaseModel):
    filename: str
    chunk_count: int
    file_hash: str
    skipped: bool = False


class BriefWorkflowResult(BaseModel):
    run_id: str
    created_at: str
    brief: LandscapeBrief
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    selected_reference_projects: list[ReferenceProject] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    artifact_path: str


class SiteWorkflowResult(BaseModel):
    run_id: str
    created_at: str
    analysis: SiteAnalysis
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    recommended_projects: list[ReferenceProject] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    artifact_path: str


class WorkflowAssets(BaseModel):
    sense_mapping: dict[str, str]
    compliance_rules: dict


class HistoryRecord(BaseModel):
    run_id: str
    workflow_type: str
    created_at: str
    project_name: str | None = None
    input_preview: str
    result_preview: dict
    artifact_path: str


def utc_now_string() -> str:
    return datetime.now().isoformat(timespec="seconds")


def path_to_string(path: Path) -> str:
    return str(path.resolve())
