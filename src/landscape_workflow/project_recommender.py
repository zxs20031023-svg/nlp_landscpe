from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .config import PATHS, load_json
from .models import ReferenceProject, SiteAnalysis


class ProjectRecommendationEngine:
    def __init__(self, library_path: Path | None = None) -> None:
        self.library_path = library_path or PATHS.project_case_library_file
        self.projects = self._load_library()

    def _load_library(self) -> list[dict]:
        library = load_json(self.library_path)
        return library.get("projects", [])

    def _save_library(self) -> None:
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.library_path.write_text(
            json.dumps({"projects": self.projects}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    @staticmethod
    def _joined_analysis_text(analysis: SiteAnalysis | None) -> str:
        if analysis is None:
            return ""
        return "\n".join(
            [
                analysis.location_context,
                analysis.climate_environment,
                analysis.topography_features,
                " ".join(analysis.opportunities),
                " ".join(analysis.constraints),
                " ".join(analysis.design_suggestions),
            ]
        )

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        return [item for item in re.split(r"[，。；、/,\s]+", query) if item]

    @staticmethod
    def _stable_project_id(prefix: str, raw_text: str) -> str:
        digest = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"

    def recommend(
        self,
        *,
        site_text: str,
        analysis: SiteAnalysis | None = None,
        top_k: int = 4,
    ) -> list[ReferenceProject]:
        query_text = f"{site_text}\n{self._joined_analysis_text(analysis)}".strip()
        normalized_query = self._normalize(query_text)
        recommendations: list[ReferenceProject] = []

        for project in self.projects:
            keywords = project.get("keywords", [])
            target_users = project.get("target_users", [])
            highlights = project.get("highlights", [])
            scene = project.get("scene", "")
            project_type = project.get("project_type", "")
            summary = project.get("summary", "")

            matched_keywords = [keyword for keyword in keywords if keyword and keyword in query_text]
            matched_users = [user for user in target_users if user and user in query_text]

            reference_text = " ".join(
                [
                    project.get("name", ""),
                    project.get("city", ""),
                    project_type,
                    scene,
                    summary,
                    " ".join(keywords),
                    " ".join(highlights),
                ]
            )
            semantic_ratio = SequenceMatcher(
                None,
                normalized_query,
                self._normalize(reference_text),
            ).ratio()

            score = len(matched_keywords) * 18 + len(matched_users) * 10 + semantic_ratio * 100
            if scene and scene in query_text:
                score += 12
            if project_type and project_type in query_text:
                score += 10

            matching_points = matched_keywords[:]
            matching_points.extend(point for point in matched_users if point not in matching_points)
            if semantic_ratio >= 0.18:
                matching_points.append("场景语义接近")

            matching_points = matching_points[:5]
            reason_parts: list[str] = []
            if matched_keywords:
                reason_parts.append(f"命中关键词：{'、'.join(matched_keywords[:4])}")
            if matched_users:
                reason_parts.append(f"匹配服务人群：{'、'.join(matched_users[:3])}")
            if semantic_ratio >= 0.18:
                reason_parts.append("整体空间场景和设计诉求较为接近")
            if not reason_parts:
                reason_parts.append("与当前场地在项目类型或设计策略上具有参考价值")

            recommendations.append(
                ReferenceProject(
                    project_id=project["project_id"],
                    name=project["name"],
                    city=project["city"],
                    project_type=project_type,
                    scene=scene,
                    summary=summary,
                    source_url=project.get("source_url", ""),
                    source_label=project.get("source_label", "本地案例库"),
                    library_source=project.get("library_source", "local"),
                    keywords=keywords,
                    target_users=target_users,
                    highlights=highlights,
                    matching_points=matching_points,
                    recommendation_reason="；".join(reason_parts),
                    similarity_score=round(min(score, 100.0), 1),
                )
            )

        recommendations.sort(key=lambda item: item.similarity_score, reverse=True)
        return recommendations[:top_k]

    def search_online_cases(self, query: str, max_results: int = 5) -> list[ReferenceProject]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise RuntimeError("缺少在线检索依赖，请先安装 requirements.txt 中新增的 ddgs 依赖。") from exc

        search_query = query.strip() or "景观设计 公园 案例"
        tokens = self._tokenize_query(search_query)
        results: list[ReferenceProject] = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                f"{search_query} 景观 公园 案例",
                region="cn-zh",
                max_results=max_results,
            )

            for rank, item in enumerate(search_results, start=1):
                title = item.get("title", "在线案例")
                body = item.get("body", "")
                url = item.get("href", "")
                matched_keywords = [token for token in tokens if token and token in f"{title} {body}"]
                base_score = max(55.0, 96.0 - rank * 8)
                score = min(100.0, base_score + len(matched_keywords) * 4)
                results.append(
                    ReferenceProject(
                        project_id=self._stable_project_id("online", f"{title}{url}"),
                        name=title[:80],
                        city="在线检索",
                        project_type="在线案例",
                        scene=search_query,
                        summary=body or "在线案例检索结果，建议打开链接查看详情。",
                        source_url=url,
                        source_label="DuckDuckGo",
                        library_source="online_imported",
                        keywords=matched_keywords or tokens[:4],
                        target_users=[],
                        highlights=["支持导入到本地案例库", "可作为任务书参考案例"],
                        matching_points=matched_keywords[:4],
                        recommendation_reason="来自在线案例检索结果，可人工甄别后导入本地案例库。",
                        similarity_score=round(score, 1),
                    )
                )

        return results

    def add_cases_to_library(
        self,
        projects: list[dict] | list[ReferenceProject],
    ) -> tuple[int, list[ReferenceProject]]:
        existing_ids = {item["project_id"] for item in self.projects}
        imported_projects: list[ReferenceProject] = []

        for project in projects:
            item = project if isinstance(project, ReferenceProject) else ReferenceProject(**project)
            if item.project_id in existing_ids:
                continue

            record = item.model_dump(mode="json")
            record["library_source"] = "local"
            if not record.get("source_label"):
                record["source_label"] = "在线导入"
            self.projects.append(record)
            existing_ids.add(item.project_id)
            imported_projects.append(ReferenceProject(**record))

        if imported_projects:
            self._save_library()

        return len(imported_projects), imported_projects
