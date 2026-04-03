from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from .config import PATHS, ensure_runtime_dirs, load_json
from .knowledge_base import KnowledgeBaseManager
from .llm import LLMWorkflowClient
from .loaders import load_documents
from .models import (
    BriefWorkflowResult,
    HistoryRecord,
    ReferenceProject,
    RetrievalHit,
    SiteWorkflowResult,
    WorkflowAssets,
    WorkflowSettings,
    path_to_string,
    utc_now_string,
)
from .persistence import save_digital_brief, save_history_record
from .project_recommender import ProjectRecommendationEngine
from .rules import ComplianceRuleEngine


class LandscapeWorkflowService:
    @staticmethod
    def load_assets() -> WorkflowAssets:
        ensure_runtime_dirs()
        return WorkflowAssets(
            sense_mapping=load_json(PATHS.sense_mapping_file),
            compliance_rules=load_json(PATHS.compliance_rules_file),
        )

    def __init__(self) -> None:
        assets = self.load_assets()
        self.assets = assets
        self.rule_engine = ComplianceRuleEngine(
            rules_config=assets.compliance_rules,
            sense_mapping=assets.sense_mapping,
        )
        self.project_recommender = ProjectRecommendationEngine()

    def run_design_brief(
        self,
        user_input: str,
        settings: WorkflowSettings,
        reference_projects: list[dict] | list[ReferenceProject] | None = None,
    ) -> BriefWorkflowResult:
        kb_manager = KnowledgeBaseManager(
            api_key=settings.api_key,
            base_url=settings.base_url,
            embedding_model=settings.embedding_model,
            persist_directory=PATHS.chroma_db,
        )
        kb_stats = kb_manager.get_stats()
        if not kb_stats.ready:
            raise ValueError("规范知识库尚未就绪，请先上传规范文件。")

        selected_reference_projects = [
            item if isinstance(item, ReferenceProject) else ReferenceProject(**item)
            for item in (reference_projects or [])
        ]

        retrieved_docs = kb_manager.retrieve(query=user_input, top_k=settings.retrieval_top_k)
        retrieval_hits = [
            RetrievalHit(
                source=doc.metadata.get("source", "未知来源"),
                excerpt=doc.page_content[:500],
            )
            for doc in retrieved_docs
        ]

        llm_client = LLMWorkflowClient(settings)
        brief, token_usage = llm_client.generate_brief(
            user_input=user_input,
            sense_mapping=self.assets.sense_mapping,
            retrieval_hits=retrieval_hits,
            reference_projects=selected_reference_projects,
        )
        brief, applied_rules = self.rule_engine.apply_to_brief(brief=brief, user_input=user_input)

        run_id = uuid.uuid4().hex[:10]
        created_at = utc_now_string()
        artifact_payload = {
            "project_name": settings.project_name,
            "run_id": run_id,
            "created_at": created_at,
            "user_input": user_input,
            "brief": brief.model_dump(mode="json"),
            "selected_reference_projects": [
                project.model_dump(mode="json") for project in selected_reference_projects
            ],
            "retrieval_hits": [item.model_dump(mode="json") for item in retrieval_hits],
            "applied_rules": [item.model_dump(mode="json") for item in applied_rules],
        }
        brief_path = save_digital_brief(
            project_name=settings.project_name,
            created_at=created_at,
            run_id=run_id,
            payload=artifact_payload,
        )

        history_record = HistoryRecord(
            run_id=run_id,
            workflow_type="design_brief",
            created_at=created_at,
            project_name=settings.project_name,
            input_preview=user_input[:200],
            result_preview={
                "brief": brief.model_dump(mode="json"),
                "reference_project_names": [project.name for project in selected_reference_projects],
            },
            artifact_path=path_to_string(brief_path),
        )
        save_history_record(history_record)

        return BriefWorkflowResult(
            run_id=run_id,
            created_at=created_at,
            brief=brief,
            retrieval_hits=retrieval_hits,
            applied_rules=applied_rules,
            selected_reference_projects=selected_reference_projects,
            token_usage=token_usage,
            artifact_path=path_to_string(brief_path),
        )

    def run_site_analysis(
        self,
        site_text: str,
        uploaded_file,
        settings: WorkflowSettings,
    ) -> SiteWorkflowResult:
        combined_site_text = site_text.strip()
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(
                prefix="landscape_site_",
                suffix=suffix,
                dir=str(PATHS.temp_dir),
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = Path(temp_file.name)
            try:
                docs = load_documents(temp_path)
                extracted_text = "\n".join(doc.page_content for doc in docs if doc.page_content)
                combined_site_text = (
                    f"【文件提取内容】\n{extracted_text}\n\n【用户补充描述】\n{site_text.strip()}"
                ).strip()
            finally:
                temp_path.unlink(missing_ok=True)

        if not combined_site_text:
            raise ValueError("请上传场地资料或输入场地描述。")

        llm_client = LLMWorkflowClient(settings)
        analysis, token_usage = llm_client.generate_site_analysis(combined_site_text)
        analysis, applied_rules = self.rule_engine.apply_to_site_analysis(
            analysis=analysis,
            site_text=combined_site_text,
        )
        recommended_projects = self.project_recommender.recommend(
            site_text=combined_site_text,
            analysis=analysis,
            top_k=4,
        )

        run_id = uuid.uuid4().hex[:10]
        created_at = utc_now_string()
        history_record = HistoryRecord(
            run_id=run_id,
            workflow_type="site_analysis",
            created_at=created_at,
            project_name=settings.project_name,
            input_preview=combined_site_text[:200],
            result_preview={
                "analysis": analysis.model_dump(mode="json"),
                "recommended_project_names": [project.name for project in recommended_projects],
            },
            artifact_path="",
        )
        history_path = save_history_record(history_record)
        history_record.artifact_path = path_to_string(history_path)
        history_path = save_history_record(history_record)

        return SiteWorkflowResult(
            run_id=run_id,
            created_at=created_at,
            analysis=analysis,
            applied_rules=applied_rules,
            recommended_projects=recommended_projects,
            token_usage=token_usage,
            artifact_path=path_to_string(history_path),
        )
