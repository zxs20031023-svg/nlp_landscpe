from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from landscape_workflow.config import (
    APP_TITLE,
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODELS,
    PATHS,
    get_recent_history,
)
from landscape_workflow.knowledge_base import KnowledgeBaseManager
from landscape_workflow.loaders import load_documents, split_documents
from landscape_workflow.models import KnowledgeBaseStats, WorkflowSettings
from landscape_workflow.service import LandscapeWorkflowService


st.set_page_config(page_title="景观设计工作流", page_icon="🌿", layout="wide")


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(234, 122, 91, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(52, 107, 87, 0.10), transparent 30%),
                    linear-gradient(180deg, #f7f4ef 0%, #f3efe7 100%);
            }
            .block-container {
                max-width: 1260px;
                padding-top: 4.8rem;
                padding-bottom: 2rem;
            }
            [data-testid="stHeader"] {
                background: rgba(247, 244, 239, 0.82);
                backdrop-filter: blur(10px);
            }
            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.82);
            }
            [data-testid="stSidebarContent"] {
                padding-top: 1.2rem;
            }
            div[data-baseweb="tab-list"] {
                gap: 0.4rem;
                padding-top: 0.2rem;
            }
            div[data-baseweb="tab"] {
                border-radius: 14px 14px 0 0;
                background: rgba(255, 255, 255, 0.7);
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .stButton > button,
            .stDownloadButton > button {
                border-radius: 12px;
                min-height: 2.8rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bootstrap_state() -> None:
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("prompt_tokens", 0)
    st.session_state.setdefault("completion_tokens", 0)
    st.session_state.setdefault("last_brief_result", None)
    st.session_state.setdefault("last_site_result", None)
    st.session_state.setdefault("last_brief_input", "")
    st.session_state.setdefault("last_site_input", "")
    st.session_state.setdefault("kb_precheck_result", None)
    st.session_state.setdefault("project_name", "课程展示项目")
    st.session_state.setdefault("site_recommendations", [])
    st.session_state.setdefault("selected_reference_project_ids", [])
    st.session_state.setdefault("online_case_results", [])
    st.session_state.setdefault("selected_online_case_ids", [])


def update_token_metrics(token_usage: dict) -> None:
    st.session_state.total_tokens += token_usage.get("total_tokens", 0)
    st.session_state.prompt_tokens += token_usage.get("prompt_tokens", 0)
    st.session_state.completion_tokens += token_usage.get("completion_tokens", 0)


def render_metric_grid(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            with st.container(border=True):
                st.metric(label=label, value=value)


def render_section_header(tag: str, title: str, body: str) -> None:
    st.caption(tag)
    st.subheader(title)
    st.write(body)


def compute_project_progress(kb_stats: KnowledgeBaseStats) -> list[dict]:
    return [
        {
            "done": True,
            "title": "配置项目与模型",
            "desc": "填写项目名称、主模型和嵌入模型后，即可进入知识库构建与结果生成阶段。",
        },
        {
            "done": kb_stats.ready,
            "title": "构建规范知识库",
            "desc": "上传规范文件或导入内置规范资源，建立后续 RAG 检索基础。",
        },
        {
            "done": st.session_state.last_brief_result is not None,
            "title": "生成数字化任务书",
            "desc": "把模糊需求转成结构化设计任务书，并附带规范依据与人工复核建议。",
        },
        {
            "done": st.session_state.last_site_result is not None,
            "title": "输出场地分析与案例推荐",
            "desc": "完成场地分析后自动推荐相似项目，可选择同步到任务书成果中。",
        },
    ]


def render_progress_panel(kb_stats: KnowledgeBaseStats) -> None:
    steps = compute_project_progress(kb_stats)
    completed = sum(1 for item in steps if item["done"])
    st.progress(completed / len(steps))
    st.caption(f"当前进度：{completed} / {len(steps)}")

    for index, item in enumerate(steps, start=1):
        status = "已完成" if item["done"] else "待完成"
        with st.container(border=True):
            st.markdown(f"**步骤 {index}｜{item['title']}**")
            st.caption(status)
            st.write(item["desc"])


def infer_brief_review(data: dict, user_input: str) -> tuple[str, list[str]]:
    retrieval_count = len(data.get("retrieval_hits", []))
    rule_count = len(data.get("applied_rules", []))
    warnings = data["brief"].get("warnings", [])

    if retrieval_count >= 2 and rule_count >= 1:
        confidence = "高"
    elif retrieval_count >= 1:
        confidence = "中"
    else:
        confidence = "低"

    review_items: list[str] = []
    if retrieval_count == 0:
        review_items.append("本次未检索到明确规范依据，建议人工复核关键约束。")
    if "儿童" in user_input and not any("儿童" in zone for zone in data["brief"]["functional_zones"]):
        review_items.append("输入提到儿童场景，但结果未明确儿童活动区，请检查功能分区。")
    if ("适老" in user_input or "轮椅" in user_input) and data["brief"]["path_slope_max_percentage"] > 5:
        review_items.append("适老或无障碍需求下坡度偏高，建议重点复核。")
    if not warnings:
        review_items.append("系统未触发显式警告，但仍建议复核植物安全、雨洪与通行条件。")
    if data["brief"]["hardscape_ratio"] > 0.45:
        review_items.append("硬质铺装比例偏高，建议确认与生态导向是否一致。")
    if data.get("selected_reference_projects"):
        review_items.append("已同步参考项目，请注意案例仅用于启发，不应替代当前场地约束判断。")
    return confidence, review_items


def infer_site_review(data: dict) -> tuple[str, list[str]]:
    rule_count = len(data.get("applied_rules", []))
    suggestions = len(data["analysis"].get("design_suggestions", []))

    if rule_count >= 2 and suggestions >= 4:
        confidence = "高"
    elif suggestions >= 3:
        confidence = "中"
    else:
        confidence = "低"

    review_items: list[str] = []
    if not data.get("applied_rules"):
        review_items.append("本次未命中场景化规则，请人工确认是否存在适老、积水、高差等隐含条件。")
    if len(data["analysis"].get("constraints", [])) < 2:
        review_items.append("约束条件偏少，建议补充交通、管线、日照与权属限制。")
    if len(data["analysis"].get("design_suggestions", [])) < 3:
        review_items.append("设计建议偏少，建议补充空间、植物和海绵设施策略。")
    if data.get("recommended_projects"):
        review_items.append("系统已给出相似项目推荐，可勾选后同步到数字化任务书。")
    return confidence, review_items


def precheck_uploaded_file(uploaded_file) -> dict:
    if uploaded_file is None:
        raise ValueError("请先选择文件。")

    suffix = Path(uploaded_file.name).suffix
    temp_path = PATHS.temp_dir / f"precheck_{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    try:
        documents = load_documents(temp_path)
        splits = split_documents(documents)
        first_preview = documents[0].page_content[:180].replace("\n", " ") if documents else ""
        return {
            "filename": uploaded_file.name,
            "document_count": len(documents),
            "chunk_count": len(splits),
            "preview": first_preview,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def resolve_spec_preview_path(filename: str, source_path: str = "") -> Path | None:
    candidates: list[Path] = []
    if source_path:
        candidates.append(Path(source_path))
    candidates.append(PATHS.knowledge_base_dir / filename)
    candidates.append(PATHS.docs_dir / "research" / filename)

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    for base_dir in [PATHS.resources_dir, PATHS.docs_dir]:
        matches = list(base_dir.rglob(filename))
        if matches:
            return matches[0]
    return None


def build_spec_preview(file_path: Path) -> dict:
    documents = load_documents(file_path)
    preview_text = "\n\n".join(
        doc.page_content.strip()
        for doc in documents[:2]
        if getattr(doc, "page_content", "").strip()
    )[:1800]
    return {
        "文件名": file_path.name,
        "文件类型": file_path.suffix.replace(".", "").upper() or "UNKNOWN",
        "解析单元": len(documents),
        "预览路径": str(file_path),
        "预览内容": preview_text or "该规范已解析，但未提取到可展示的预览文本。",
    }


def build_spec_catalog(
    kb_manager: KnowledgeBaseManager,
    kb_stats: KnowledgeBaseStats,
) -> list[dict]:
    indexed_records = {item.filename: item for item in kb_stats.documents}
    catalog: list[dict] = []
    seen_names: set[str] = set()

    for file_path in kb_manager.list_preferred_bundled_files():
        record = indexed_records.get(file_path.name)
        catalog.append(
            {
                "filename": file_path.name,
                "source_type": "内置规范",
                "ingest_status": "已入库" if record else "未入库",
                "file_type": file_path.suffix.replace(".", "").upper() or "UNKNOWN",
                "chunk_count": record.chunk_count if record else 0,
                "ingested_at": record.ingested_at if record else "-",
                "preview_path": str(file_path),
                "source_path": str(file_path),
                "file_hash": record.file_hash if record else "-",
            }
        )
        seen_names.add(file_path.name)

    for record in kb_stats.documents:
        if record.filename in seen_names:
            continue
        preview_path = resolve_spec_preview_path(record.filename, record.source_path)
        catalog.append(
            {
                "filename": record.filename,
                "source_type": "外部规范",
                "ingest_status": "已入库",
                "file_type": Path(record.filename).suffix.replace(".", "").upper() or "UNKNOWN",
                "chunk_count": record.chunk_count,
                "ingested_at": record.ingested_at,
                "preview_path": str(preview_path) if preview_path else "",
                "source_path": record.source_path,
                "file_hash": record.file_hash,
            }
        )

    catalog.sort(key=lambda item: (item["source_type"] != "内置规范", item["filename"].lower()))
    return catalog


def render_spec_catalog(catalog: list[dict]) -> None:
    if not catalog:
        st.info("当前还没有可展示的规范资源。")
        return

    st.markdown("### 规范清单")
    for item in catalog:
        with st.container(border=True):
            st.json(
                {
                    "规范名称": item["filename"],
                    "规范来源": item["source_type"],
                    "入库状态": item["ingest_status"],
                    "文件类型": item["file_type"],
                    "切分片段数": item["chunk_count"],
                    "最近入库时间": item["ingested_at"],
                    "预览路径": item["preview_path"] or "暂无可预览源文件",
                },
                expanded=False,
            )


def render_spec_preview_panel(catalog: list[dict]) -> None:
    with st.container(border=True):
        st.markdown("### 规范内容预览")
        if not catalog:
            st.info("当前没有可预览的规范。")
            return

        option_map = {
            f"{item['filename']}｜{item['source_type']}｜{item['ingest_status']}": item
            for item in catalog
        }
        selected_label = st.selectbox(
            "选择要预览的规范",
            options=list(option_map.keys()),
            key="kb_preview_selector",
        )
        selected_item = option_map[selected_label]
        preview_path = resolve_spec_preview_path(
            selected_item["filename"],
            selected_item.get("preview_path", "") or selected_item.get("source_path", ""),
        )

        st.json(
            {
                "规范名称": selected_item["filename"],
                "规范来源": selected_item["source_type"],
                "入库状态": selected_item["ingest_status"],
                "文件类型": selected_item["file_type"],
                "切分片段数": selected_item["chunk_count"],
                "最近入库时间": selected_item["ingested_at"],
            },
            expanded=False,
        )

        if not preview_path:
            st.warning("当前规范没有可直接访问的原始文件，因此无法展示内容预览。")
            return

        try:
            preview_info = build_spec_preview(preview_path)
            st.text_area(
                "规范正文预览",
                value=preview_info["预览内容"],
                height=260,
                disabled=True,
            )
        except Exception as exc:
            st.error(f"规范预览失败：{exc}")


def sync_recommendation_state(recommendations: list[dict], reset_to_default: bool = False) -> None:
    st.session_state.site_recommendations = recommendations
    current_ids = [item["project_id"] for item in recommendations]
    previous_ids = st.session_state.get("selected_reference_project_ids", [])

    if reset_to_default or not previous_ids:
        st.session_state.selected_reference_project_ids = current_ids[:2]
        return

    filtered_ids = [project_id for project_id in previous_ids if project_id in current_ids]
    st.session_state.selected_reference_project_ids = filtered_ids or current_ids[:2]


def get_selected_reference_projects() -> list[dict]:
    selected_ids = set(st.session_state.get("selected_reference_project_ids", []))
    recommendations = st.session_state.get("site_recommendations", [])
    return [item for item in recommendations if item["project_id"] in selected_ids]


def merge_recommendations(base: list[dict], extra: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {item["project_id"]: item for item in base}
    for item in extra:
        merged[item["project_id"]] = item
    return sorted(merged.values(), key=lambda item: item.get("similarity_score", 0), reverse=True)


def render_reference_project_cards(recommendations: list[dict], selected_ids: set[str]) -> None:
    for project in recommendations:
        with st.container(border=True):
            left, right = st.columns([0.26, 0.74])
            with left:
                st.metric("匹配度", f"{project['similarity_score']:.1f}")
                st.caption("已纳入任务书" if project["project_id"] in selected_ids else "未纳入任务书")
            with right:
                st.markdown(f"**{project['name']}**")
                st.write(f"{project['city']}｜{project['project_type']}｜{project['scene']}")
                st.write(project["summary"])
                if project.get("source_url"):
                    st.markdown(f"[查看来源]({project['source_url']})")
                else:
                    st.caption(f"来源：{project.get('source_label', '本地案例库')}")
                if project.get("matching_points"):
                    st.caption(f"匹配点：{'、'.join(project['matching_points'])}")
                st.write(f"推荐理由：{project['recommendation_reason']}")
                if project.get("highlights"):
                    st.write(f"可借鉴亮点：{'；'.join(project['highlights'])}")


def render_reference_project_selector(recommendations: list[dict]) -> None:
    if not recommendations:
        st.info("暂未生成相关项目推荐。")
        return

    selected_ids = st.session_state.get("selected_reference_project_ids", [])
    option_map = {
        f"{project['name']}｜{project['city']}｜匹配度 {project['similarity_score']:.1f}": project["project_id"]
        for project in recommendations
    }
    inverse_map = {value: key for key, value in option_map.items()}
    default_options = [inverse_map[item] for item in selected_ids if item in inverse_map]
    chosen_options = st.multiselect(
        "选择要同步到数字化任务书的参考项目",
        options=list(option_map.keys()),
        default=default_options,
        help="勾选后的项目会在下一次生成数字化任务书时自动写入成果文件。",
    )
    st.session_state.selected_reference_project_ids = [option_map[item] for item in chosen_options]
    st.caption(f"当前已选择 {len(st.session_state.selected_reference_project_ids)} 个参考项目。")
    render_reference_project_cards(
        recommendations=recommendations,
        selected_ids=set(st.session_state.selected_reference_project_ids),
    )


def render_selected_reference_summary() -> None:
    selected_projects = get_selected_reference_projects()
    with st.container(border=True):
        st.markdown("### 已同步的参考项目")
        if not selected_projects:
            st.info("当前还没有选中的参考项目。先完成场地分析后，系统会自动给出相似案例推荐。")
            return
        for project in selected_projects:
            st.write(
                f"- {project['name']}｜{project['city']}｜{project['project_type']}｜匹配度 {project['similarity_score']:.1f}"
            )
        st.caption("如需调整，请回到“场地分析”页修改勾选结果。")


def render_online_case_search(service: LandscapeWorkflowService, default_query: str) -> None:
    with st.container(border=True):
        st.markdown("### 在线案例检索")
        online_query = st.text_input(
            "在线检索词",
            value=default_query,
            key="online_case_query",
            help="建议输入城市、场地类型、设计主题或目标人群，例如“深圳 社区口袋公园 适老 儿童活动”。",
        )
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            search_btn = st.button("检索在线案例", use_container_width=True)
        with action_col2:
            import_btn = st.button("导入所选在线案例到案例库", use_container_width=True)

        if search_btn:
            with st.spinner("正在检索在线案例..."):
                try:
                    results = service.project_recommender.search_online_cases(online_query, max_results=6)
                    st.session_state.online_case_results = [item.model_dump(mode="json") for item in results]
                    st.session_state.selected_online_case_ids = [
                        item["project_id"] for item in st.session_state.online_case_results[:2]
                    ]
                    st.success(f"已检索到 {len(results)} 条在线案例。")
                except Exception as exc:
                    st.error(f"在线检索失败：{exc}")

        online_results = st.session_state.get("online_case_results", [])
        if online_results:
            option_map = {
                f"{item['name']}｜匹配度 {item['similarity_score']:.1f}": item["project_id"]
                for item in online_results
            }
            inverse_map = {value: key for key, value in option_map.items()}
            default_options = [
                inverse_map[item]
                for item in st.session_state.get("selected_online_case_ids", [])
                if item in inverse_map
            ]
            chosen_options = st.multiselect(
                "选择要导入本地案例库的在线案例",
                options=list(option_map.keys()),
                default=default_options,
                key="online_case_selector",
            )
            st.session_state.selected_online_case_ids = [option_map[item] for item in chosen_options]
            render_reference_project_cards(
                recommendations=online_results,
                selected_ids=set(st.session_state.selected_online_case_ids),
            )

        if import_btn:
            online_results = st.session_state.get("online_case_results", [])
            selected_ids = set(st.session_state.get("selected_online_case_ids", []))
            selected_projects = [item for item in online_results if item["project_id"] in selected_ids]
            if not selected_projects:
                st.warning("请先勾选要导入的在线案例。")
            else:
                imported_count, imported_projects = service.project_recommender.add_cases_to_library(selected_projects)
                if imported_count == 0:
                    st.info("所选在线案例已存在于本地案例库中，无需重复导入。")
                else:
                    imported_payload = [item.model_dump(mode="json") for item in imported_projects]
                    merged_recommendations = merge_recommendations(
                        st.session_state.get("site_recommendations", []),
                        imported_payload,
                    )
                    sync_recommendation_state(merged_recommendations)
                    current_selected = set(st.session_state.get("selected_reference_project_ids", []))
                    current_selected.update(item["project_id"] for item in imported_payload)
                    st.session_state.selected_reference_project_ids = list(current_selected)
                    st.success(f"已导入 {imported_count} 条在线案例到本地案例库，并可直接同步到任务书。")


def render_sidebar(service: LandscapeWorkflowService) -> tuple[WorkflowSettings, KnowledgeBaseStats]:
    with st.sidebar:
        st.title("项目控制台")
        project_name = st.text_input("项目名称", value=st.session_state.project_name, key="project_name")
        st.caption("建议使用真实地块、课程或方案名称，方便后续检索历史记录。")

        with st.expander("模型配置", expanded=True):
            api_key = st.text_input("API Key", type="password", placeholder="sk-...")
            base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
            selected_model = st.selectbox("主模型", DEFAULT_MODELS)
            model_name = (
                st.text_input("自定义模型名称", value="qwen-plus")
                if selected_model == "自定义输入..."
                else selected_model
            )
            embedding_model = st.text_input("嵌入模型", value=DEFAULT_EMBEDDING_MODEL)
            retrieval_top_k = st.slider("规范检索条数", min_value=1, max_value=8, value=4)

        st.markdown("### 使用状态")
        st.metric("累计 Token", f"{st.session_state.total_tokens:,}")
        st.metric("提示词 Token", f"{st.session_state.prompt_tokens:,}")
        st.metric("生成 Token", f"{st.session_state.completion_tokens:,}")

        kb_manager = KnowledgeBaseManager(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            persist_directory=PATHS.chroma_db,
        )
        kb_stats = kb_manager.get_stats()

        st.markdown("### 知识库状态")
        if kb_stats.ready:
            st.success(f"知识库已就绪，共 {kb_stats.document_count} 个文件")
        else:
            st.warning("知识库尚未初始化")

        st.caption(f"向量库目录：{PATHS.chroma_db}")

        if kb_stats.documents:
            with st.expander("已登记规范文件"):
                for item in kb_stats.documents:
                    st.write(f"- {item.filename}｜{item.chunk_count} 个片段")

        with st.expander("语义映射样例"):
            sample_mapping = dict(list(service.assets.sense_mapping.items())[:8])
            st.json(sample_mapping)

        with st.expander("首次使用建议"):
            st.write("1. 先填写 API Key 和模型参数。")
            st.write("2. 优先导入内置规范库，或上传 DOCX / TXT 文件。")
            st.write("3. 完成一次场地分析，系统会自动给出相似项目推荐。")
            st.write("4. 勾选参考项目后再生成数字化任务书。")

    return (
        WorkflowSettings(
            project_name=project_name,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            embedding_model=embedding_model,
            retrieval_top_k=retrieval_top_k,
        ),
        kb_stats,
    )


def render_overview_tab(settings: WorkflowSettings, kb_stats: KnowledgeBaseStats) -> None:
    history = get_recent_history(limit=8)
    project_history = [item for item in history if item.get("project_name") == settings.project_name]

    st.title(APP_TITLE)
    st.caption("围绕规范入库、需求转译、场地分析和案例参考，构建一个更适合课程展示与专业前期分析的工作台。")

    render_metric_grid(
        [
            ("当前项目", settings.project_name),
            ("知识库文件", str(kb_stats.document_count)),
            ("任务书结果", "1" if st.session_state.last_brief_result else "0"),
            ("案例推荐数", str(len(st.session_state.get("site_recommendations", [])))),
        ]
    )

    left, right = st.columns([1.25, 1])
    with left:
        with st.container(border=True):
            st.markdown("### 项目进度")
            render_progress_panel(kb_stats)
    with right:
        with st.container(border=True):
            st.markdown("### 快速开始")
            st.write("1. 配置模型参数。")
            st.write("2. 导入规范库。")
            st.write("3. 先生成场地分析和案例推荐。")
            st.write("4. 勾选案例后生成数字化任务书。")
        with st.container(border=True):
            st.markdown("### 本轮优化亮点")
            st.write("- 顶部导航区域避让修正")
            st.write("- 场地相似项目自动推荐")
            st.write("- 参考项目可勾选同步到任务书")
            st.write("- 数字化任务书固定目录归档")

    st.markdown("### 项目历史")
    if project_history:
        for item in project_history[:5]:
            with st.expander(f"{item.get('workflow_type')}｜{item.get('created_at')}"):
                st.write(f"项目：{item.get('project_name', '默认项目')}")
                st.write(f"输入摘要：{item.get('input_preview', '')}")
                st.write(f"产物路径：{item.get('artifact_path', '')}")
                st.json(item.get("result_preview", {}))
    else:
        st.info("当前项目下还没有生成记录，可以先导入规范或运行一次场地分析。")


def render_knowledge_base_tab(settings: WorkflowSettings, kb_stats: KnowledgeBaseStats) -> None:
    render_section_header(
        "Knowledge Base",
        "规范知识库管理",
        "支持导入内置规范资源，也支持上传 TXT / PDF / DOCX / DOC 文件。建议优先使用 DOCX 或 TXT，以获得更稳定的解析效果。",
    )

    kb_manager = KnowledgeBaseManager(
        api_key=settings.api_key,
        base_url=settings.base_url,
        embedding_model=settings.embedding_model,
        persist_directory=PATHS.chroma_db,
    )
    spec_catalog = build_spec_catalog(kb_manager, kb_stats)

    left, right = st.columns([1.4, 1])
    with left:
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "上传规范文件",
                type=["txt", "pdf", "docx", "doc"],
                help="支持 TXT / PDF / DOCX / DOC",
                key="kb_upload",
            )
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                preview_btn = st.button("预检查文件", use_container_width=True)
            with action_col2:
                builtin_btn = st.button("导入内置规范", use_container_width=True)
            with action_col3:
                ingest_btn = st.button("构建或更新知识库", use_container_width=True, type="primary")

        if preview_btn:
            try:
                st.session_state.kb_precheck_result = precheck_uploaded_file(uploaded_file)
            except Exception as exc:
                st.session_state.kb_precheck_result = {"error": str(exc)}

        if builtin_btn:
            if not settings.api_key:
                st.error("请先填写 API Key。")
            else:
                with st.spinner("正在导入内置规范资源..."):
                    try:
                        results = kb_manager.ingest_bundled_resources()
                        added = [item for item in results if not item.skipped]
                        skipped = [item for item in results if item.skipped]
                        st.success(f"导入完成：新增 {len(added)} 个文件，跳过 {len(skipped)} 个重复文件。")
                    except Exception as exc:
                        st.error(f"导入失败：{exc}")

        if ingest_btn:
            if uploaded_file is None:
                st.warning("请先选择规范文件。")
            elif not settings.api_key:
                st.error("请先填写 API Key。")
            else:
                stage = st.empty()
                with st.spinner("正在构建知识库..."):
                    try:
                        stage.info("阶段 1/3：接收文件并进行基础校验")
                        stage.info("阶段 2/3：解析文档并切分片段")
                        result = kb_manager.ingest_uploaded_file(uploaded_file)
                        stage.info("阶段 3/3：写入向量库并更新清单")
                        if result.skipped:
                            st.info(f"文件已存在，已跳过重复构建：{result.filename}")
                        else:
                            st.success(f"知识库更新成功：{result.filename}，新增 {result.chunk_count} 个片段。")
                    except Exception as exc:
                        st.error(f"知识库更新失败：{exc}")
                    finally:
                        stage.empty()

        precheck = st.session_state.kb_precheck_result
        if precheck:
            with st.container(border=True):
                st.markdown("### 预检查结果")
                if precheck.get("error"):
                    st.error(f"文件预检查失败：{precheck['error']}")
                else:
                    render_metric_grid(
                        [
                            ("文件名", precheck["filename"]),
                            ("解析单元", str(precheck["document_count"])),
                            ("预计片段", str(precheck["chunk_count"])),
                        ]
                    )
                    st.text_area("内容预览", value=precheck["preview"], height=120, disabled=True)

    with right:
        render_spec_catalog(spec_catalog)
        render_spec_preview_panel(spec_catalog)

        with st.container(border=True):
            st.markdown("### 上传建议")
            st.write("- 优先上传可复制文本的 DOCX / TXT。")
            st.write("- 扫描版 PDF 建议同步准备 OCR 文本。")
            st.write("- 首次入库建议从 1 到 2 个核心规范文件开始。")
            st.write(f"- 当前知识库状态：{'已就绪' if kb_stats.ready else '未就绪'}")


def render_brief_tab(settings: WorkflowSettings, service: LandscapeWorkflowService) -> None:
    render_section_header(
        "Design Brief",
        "设计需求转译",
        "将自然语言需求转成结构化数字化任务书，并把你选中的参考案例一起写入最终成果文件。",
    )

    sample_options = {
        "适老社区口袋公园": "设计一个适老化社区口袋公园，需要安静、生态、便于轮椅通行，希望保留一定林荫，并设置儿童活动区。儿童区附近不要种植夹竹桃。",
        "街角微更新绿地": "设计一个社区街角微更新绿地，要求兼顾社交、亲子和夜间安全，尽量提升绿量并保留一定活动铺装。",
        "滨水康养步道": "设计一个滨水康养步道和休憩节点系统，强调疗愈、亲水、适老与夜间照明安全。",
    }

    col1, col2 = st.columns([1.05, 1])
    with col1:
        with st.container(border=True):
            sample_key = st.selectbox("快速示例", list(sample_options.keys()))
            user_input = st.text_area("输入设计需求", value=sample_options[sample_key], height=250)
            st.caption("建议输入项目类型、风格倾向、服务人群、核心功能和敏感限制条件。")
            st.caption(f"任务书固定归档目录：{PATHS.digital_briefs_dir}")
            generate = st.button("生成数字化任务书", use_container_width=True, type="primary")

        render_selected_reference_summary()

    with col2:
        data = st.session_state.last_brief_result
        review_input = st.session_state.last_brief_input or user_input
        if generate:
            if not settings.api_key:
                st.error("请先填写 API Key。")
            elif not user_input.strip():
                st.warning("请输入设计需求。")
            else:
                selected_reference_projects = get_selected_reference_projects()
                with st.spinner("正在执行需求转译工作流..."):
                    try:
                        result = service.run_design_brief(
                            user_input=user_input,
                            settings=settings,
                            reference_projects=selected_reference_projects,
                        )
                        st.session_state.last_brief_result = result.model_dump(mode="json")
                        st.session_state.last_brief_input = user_input
                        data = st.session_state.last_brief_result
                        review_input = user_input
                        update_token_metrics(result.token_usage)
                        st.success("数字化任务书已生成并归档。")
                    except Exception as exc:
                        st.error(f"生成失败：{exc}")

        if not data:
            st.info("选择一个示例或输入真实需求后，点击按钮开始生成。")
            return

        brief = data["brief"]
        confidence, review_items = infer_brief_review(data, review_input)
        render_metric_grid(
            [
                ("项目类型", brief["project_type"]),
                ("风格偏好", brief["style_preference"]),
                ("规范依据", str(len(data["retrieval_hits"]))),
                ("参考项目", str(len(data.get("selected_reference_projects", [])))),
            ]
        )

        with st.container(border=True):
            st.markdown("### 固定归档位置")
            st.code(data["artifact_path"])

        if brief["warnings"]:
            with st.container(border=True):
                st.markdown("### 风险与修正提示")
                for warning in brief["warnings"]:
                    st.warning(warning)
        else:
            st.success("本次未触发显式风险修正。")

        with st.expander("任务书主体 JSON", expanded=True):
            st.json(brief)

        with st.expander("完整任务书档案 JSON"):
            st.json(data)
            st.download_button(
                "下载完整任务书档案 JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"{settings.project_name}_digital_landscape_brief_full.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("已同步参考项目", expanded=True):
            if data.get("selected_reference_projects"):
                render_reference_project_cards(
                    recommendations=data["selected_reference_projects"],
                    selected_ids={project["project_id"] for project in data["selected_reference_projects"]},
                )
            else:
                st.info("本次任务书未同步参考项目。")

        with st.expander("检索依据与规则命中"):
            st.markdown("**规范检索命中**")
            if data["retrieval_hits"]:
                for hit in data["retrieval_hits"]:
                    st.write(f"- {hit['source']}：{hit['excerpt'][:240]}")
            else:
                st.info("本次没有检索到有效规范片段。")

            st.markdown("**显式规则命中**")
            if data["applied_rules"]:
                for rule in data["applied_rules"]:
                    st.write(f"- {rule['title']}：{rule['warning']}")
            else:
                st.info("本次没有触发显式规则。")

        with st.expander("可信度与待人工复核项"):
            st.write(f"可信度评估：`{confidence}`")
            for item in review_items:
                st.write(f"- {item}")


def render_site_tab(settings: WorkflowSettings, service: LandscapeWorkflowService) -> None:
    render_section_header(
        "Site Analysis",
        "场地现状多维解析",
        "完成场地分析后，系统会自动从本地案例库中检索相似项目，并支持勾选同步到数字化任务书中。",
    )

    col1, col2 = st.columns([1.05, 1])
    with col1:
        with st.container(border=True):
            site_file = st.file_uploader(
                "上传场地资料",
                type=["txt", "pdf", "docx", "doc"],
                key="site_upload",
            )
            default_text = (
                "场地位于高密度老旧社区内部，周边以居住功能为主。"
                "内部高差约 3 米，存在低洼积水点，老年人口占比较高，活动空间不足。"
            )
            site_text = st.text_area("输入场地描述", value=default_text, height=240)
            st.caption("建议补充区位、周边功能、人群结构、地形水文、日照风环境和改造诉求。")
            analyze = st.button("生成场地分析报告", use_container_width=True, type="primary")

    with col2:
        data = st.session_state.last_site_result
        if analyze:
            if not settings.api_key:
                st.error("请先填写 API Key。")
            elif not site_text.strip() and site_file is None:
                st.warning("请上传文件或输入场地描述。")
            else:
                with st.spinner("正在执行场地分析工作流..."):
                    try:
                        result = service.run_site_analysis(
                            site_text=site_text,
                            uploaded_file=site_file,
                            settings=settings,
                        )
                        st.session_state.last_site_result = result.model_dump(mode="json")
                        st.session_state.last_site_input = site_text
                        data = st.session_state.last_site_result
                        update_token_metrics(result.token_usage)
                        sync_recommendation_state(data.get("recommended_projects", []), reset_to_default=True)
                    except Exception as exc:
                        st.error(f"分析失败：{exc}")

        if not data:
            st.info("上传文件或填写场地描述后，点击按钮开始分析。")
        else:
            if data.get("recommended_projects"):
                sync_recommendation_state(data.get("recommended_projects", []))

            confidence, review_items = infer_site_review(data)
            render_metric_grid(
                [
                    ("机会点", str(len(data["analysis"]["opportunities"]))),
                    ("约束条件", str(len(data["analysis"]["constraints"]))),
                    ("设计建议", str(len(data["analysis"]["design_suggestions"]))),
                    ("推荐项目", str(len(data.get("recommended_projects", [])))),
                ]
            )

            with st.expander("结构化场地分析 JSON", expanded=True):
                st.json(data["analysis"])
                st.download_button(
                    "下载场地分析 JSON",
                    data=json.dumps(data["analysis"], ensure_ascii=False, indent=2),
                    file_name=f"{settings.project_name}_site_analysis_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with st.expander("规则命中与自动补充建议"):
                if data["applied_rules"]:
                    for rule in data["applied_rules"]:
                        st.write(f"- {rule['title']}：{rule['warning']}")
                else:
                    st.info("本次没有触发额外规则。")

            with st.expander("可信度与待人工复核项"):
                st.write(f"可信度评估：`{confidence}`")
                for item in review_items:
                    st.write(f"- {item}")

    st.markdown("### 相关项目推荐")
    recommendations = data.get("recommended_projects", []) if data else []
    if recommendations:
        render_reference_project_selector(recommendations)
        if st.session_state.get("selected_reference_project_ids"):
            st.success("当前勾选的参考项目将在下一次生成数字化任务书时自动同步。")
    else:
        st.info("完成场地分析后，这里会出现相似项目推荐。")

    default_online_query = site_text.strip() if "site_text" in locals() else st.session_state.get("last_site_input", "")
    render_online_case_search(service, default_online_query[:80])


def render_history_tab(settings: WorkflowSettings) -> None:
    render_section_header(
        "Project Assets",
        "项目历史与结果资产",
        "历史结果不仅是日志，也可作为后续方案迭代、课程展示和成果归档的项目资产。",
    )
    history = get_recent_history(limit=20)
    project_history = [item for item in history if item.get("project_name") == settings.project_name]

    if not project_history:
        st.info("当前项目下还没有历史记录。")
        return

    render_metric_grid(
        [
            ("当前项目", settings.project_name),
            ("历史记录", str(len(project_history))),
            ("任务书次数", str(sum(1 for item in project_history if item.get("workflow_type") == "design_brief"))),
            ("场地分析次数", str(sum(1 for item in project_history if item.get("workflow_type") == "site_analysis"))),
        ]
    )

    for item in project_history:
        with st.expander(f"{item.get('workflow_type')}｜{item.get('created_at')}｜{item.get('run_id')}"):
            st.write(f"输入摘要：{item.get('input_preview', '')}")
            st.write(f"产物文件：{item.get('artifact_path', '')}")
            st.json(item.get("result_preview", {}))


def main() -> None:
    bootstrap_state()
    inject_global_styles()
    service = LandscapeWorkflowService()

    settings, kb_stats = render_sidebar(service)

    tabs = st.tabs(["工作台总览", "知识库管理", "设计任务书", "场地分析", "项目历史"])
    with tabs[0]:
        render_overview_tab(settings, kb_stats)
    with tabs[1]:
        render_knowledge_base_tab(settings, kb_stats)
    with tabs[2]:
        render_brief_tab(settings, service)
    with tabs[3]:
        render_site_tab(settings, service)
    with tabs[4]:
        render_history_tab(settings)


if __name__ == "__main__":
    main()
