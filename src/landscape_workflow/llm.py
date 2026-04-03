from __future__ import annotations

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .models import LandscapeBrief, ReferenceProject, RetrievalHit, SiteAnalysis, WorkflowSettings


class LLMWorkflowClient:
    def __init__(self, settings: WorkflowSettings) -> None:
        self.settings = settings

    def _llm(self, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or None,
            model=self.settings.model_name,
            temperature=temperature,
        )

    @staticmethod
    def _token_usage(callback) -> dict:
        return {
            "total_tokens": getattr(callback, "total_tokens", 0),
            "prompt_tokens": getattr(callback, "prompt_tokens", 0),
            "completion_tokens": getattr(callback, "completion_tokens", 0),
        }

    @staticmethod
    def _reference_projects_text(reference_projects: list[ReferenceProject]) -> str:
        if not reference_projects:
            return "无参考项目。"

        return "\n\n".join(
            [
                "\n".join(
                    [
                        f"案例名称：{project.name}",
                        f"城市：{project.city}",
                        f"项目类型：{project.project_type}",
                        f"适用场景：{project.scene}",
                        f"核心亮点：{'；'.join(project.highlights)}",
                        f"推荐原因：{project.recommendation_reason}",
                    ]
                )
                for project in reference_projects
            ]
        )

    def generate_brief(
        self,
        user_input: str,
        sense_mapping: dict[str, str],
        retrieval_hits: list[RetrievalHit],
        reference_projects: list[ReferenceProject] | None = None,
    ) -> tuple[LandscapeBrief, dict]:
        parser = JsonOutputParser(pydantic_object=LandscapeBrief)
        regulations = "\n\n".join(f"[{hit.source}]\n{hit.excerpt}" for hit in retrieval_hits) or "未检索到相关规范，请根据常见公园设计规范谨慎生成。"
        reference_project_text = self._reference_projects_text(reference_projects or [])

        prompt = PromptTemplate(
            template="""
你是一名资深风景园林设计师与合规审查顾问。请根据用户需求、语义映射规则、检索到的规范条文和参考案例，输出一个严格符合 JSON 结构的景观数字化任务书。

用户需求：
{user_input}

语义映射规则：
{sense_mapping}

规范条文：
{regulations}

参考案例：
{reference_projects}

输出要求：
1. 提取项目类型、风格倾向和功能分区。
2. 将结果量化为 canopy_closure、path_slope_max_percentage、hardscape_ratio。
3. 若发现明显冲突，可以给出 warnings，但不要省略 JSON 结构。
4. 参考案例只作为类比启发，不能照搬其数值和结论，仍需以当前场地需求和规范约束为准。
5. 不要输出 JSON 以外的解释。
{format_instructions}
""".strip(),
            input_variables=["user_input", "sense_mapping", "regulations", "reference_projects"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self._llm(temperature=0.0) | parser
        with get_openai_callback() as callback:
            result = chain.invoke(
                {
                    "user_input": user_input,
                    "sense_mapping": sense_mapping,
                    "regulations": regulations,
                    "reference_projects": reference_project_text,
                }
            )
        return LandscapeBrief(**result), self._token_usage(callback)

    def generate_site_analysis(self, site_text: str) -> tuple[SiteAnalysis, dict]:
        parser = JsonOutputParser(pydantic_object=SiteAnalysis)
        prompt = PromptTemplate(
            template="""
你是一名专业的景观规划与场地分析顾问。请依据以下场地资料，输出结构化场地分析 JSON。

场地资料：
{site_text}

输出要求：
1. 提炼区位环境、气候环境、地形水文。
2. 明确 opportunities 和 constraints。
3. 给出可执行的景观设计建议。
4. 不要输出 JSON 以外的解释。
{format_instructions}
""".strip(),
            input_variables=["site_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self._llm(temperature=0.2) | parser
        with get_openai_callback() as callback:
            result = chain.invoke({"site_text": site_text})
        return SiteAnalysis(**result), self._token_usage(callback)
