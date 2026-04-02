import streamlit as st
import os
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# --- RAG 相关核心库 ---
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 定义数据结构 (Pydantic) 
# ==========================================
# 1.1 数字化任务书输出结构 (原功能)
class LandscapeBrief(BaseModel):
    project_type: str = Field(description="项目类型，如城市公园、社区口袋公园等")
    style_preference: str = Field(description="景观风格，如现代自然、新中式等")
    canopy_closure: float = Field(description="植物郁闭度 (0.0-1.0)")
    path_slope_max_percentage: float = Field(description="最大园路坡度 (%)")
    hardscape_ratio: float = Field(description="硬质铺装比例 (0.0-1.0)")
    functional_zones: list[str] = Field(description="功能分区列表")
    warnings: list[str] = Field(description="【核心】规范冲突与修正警告说明。如果没有冲突则为空列表。")

# 1.2 场地多维解析输出结构 (新增功能)
class SiteAnalysis(BaseModel):
    location_context: str = Field(description="区位与周边环境特征总结")
    climate_environment: str = Field(description="气候条件与微环境特征总结")
    topography_features: str = Field(description="地形地貌与水文特征总结")
    opportunities: list[str] = Field(description="场地具备的开发优势与机遇 (Opportunities)")
    constraints: list[str] = Field(description="场地面临的限制因素与挑战 (Constraints)")
    design_suggestions: list[str] = Field(description="基于场地的专业初步设计建议")

# ==========================================
# 2. 模拟本地知识库 (字典/JSON)
# ==========================================
SENSE_MAPPING = {
    "幽静": "郁闭度应大于0.7，硬质铺装比例小于0.2，以自然绿化为主。",
    "开阔": "郁闭度应小于0.3，强调视线通廊，硬质铺装可适当增加。",
    "活力": "需包含集散广场、运动健身区等功能，铺装比例不低于0.4。",
    "适老": "需包含康体健身区、静态休憩区，需重点关注无障碍通行、防滑处理。",
    "生态": "绿地率需大于70%，优先使用乡土树种，水体驳岸采用软质生态驳岸。",
    "现代": "硬质铺装以简洁的几何线条为主，植物配置强调阵列感或大色块对比。"
}

# ==========================================
# 3. 核心处理流程引擎
# ==========================================

# 3.1 大模型转译链 + RAG检索引擎 (原功能)
def generate_digital_brief(api_key, base_url, model_name, user_input):
    llm = ChatOpenAI(api_key=api_key, base_url=base_url if base_url else None, model=model_name, temperature=0)
    embeddings = OpenAIEmbeddings(
        api_key=api_key, base_url=base_url if base_url else None,
        model="text-embedding-v3", chunk_size=10, check_embedding_ctx_length=False 
    )

    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        raise Exception("未找到向量数据库！请先在左侧边栏上传规范文件，系统会自动构建。")
        
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(user_input)
    retrieved_regulations = "\n".join([doc.page_content for doc in relevant_docs])

    parser = JsonOutputParser(pydantic_object=LandscapeBrief)
    prompt = PromptTemplate(
        template="""你是一个资深的风景园林架构师与合规审查专家。
请根据以下【用户输入的设计需求】，结合【本地经验映射库】和【强制性国家规范】，生成一份参数化的景观设计任务书。

【用户输入的设计需求】\n{user_input}\n
【本地经验映射库参考】\n{sense_mapping}\n
【检索到的国家规范 (强制约束)】\n{regulations}\n

【处理逻辑与要求】
1. 语义提取：理解用户的感性需求，参考映射库转化为初步物理参数。
2. 合规校验（关键）：将初步参数与【检索到的国家规范】进行比对。
3. 自动修正：如果用户需求或初步参数违反了国家规范（例如坡度超标、植物违规），你必须强制修正为合规参数，并在 warnings 中详细记录“因什么规范，将什么参数或设施修改为了什么”。

请严格按照以下 JSON 格式输出：\n{format_instructions}
""",
        input_variables=["user_input", "sense_mapping", "regulations"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    mapping_str = json.dumps(SENSE_MAPPING, ensure_ascii=False, indent=2)

    with get_openai_callback() as cb:
        result = chain.invoke({
            "user_input": user_input,
            "sense_mapping": mapping_str,
            "regulations": retrieved_regulations if retrieved_regulations else "未检索到相关规范。"
        })
    return result, cb

# 3.2 场地多维智能解析引擎 (新增功能)
def generate_site_analysis(api_key, base_url, model_name, site_info):
    llm = ChatOpenAI(api_key=api_key, base_url=base_url if base_url else None, model=model_name, temperature=0.3)
    
    parser = JsonOutputParser(pydantic_object=SiteAnalysis)
    prompt = PromptTemplate(
        template="""你是一个专业的风景园林师与场地规划专家。
请根据以下【场地勘察原始资料/描述】，进行深度挖掘与多维度解析，输出结构化的场地分析报告。

【场地资料】
{site_info}

【解析要求】
1. 提炼区位环境、气候、地形地貌与水文特征。
2. 深度剖析场地具备的优势与机遇（Opportunities），以及限制条件与挑战（Constraints）。
3. 基于这些特征，给出具有实操性的专业景观设计建议。

请严格按照以下 JSON 格式输出：\n{format_instructions}
""",
        input_variables=["site_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    with get_openai_callback() as cb:
        result = chain.invoke({"site_info": site_info})
    return result, cb

# ==========================================
# 4. Streamlit 前端界面设计
# ==========================================
st.set_page_config(page_title="景观 NLP 转译系统", page_icon="🌿", layout="wide")

st.title("🌿 基于NLP的城市公园景观设计需求转译与场地解析系统")

# 初始化 Session State
if 'total_tokens' not in st.session_state: st.session_state.total_tokens = 0
if 'prompt_tokens' not in st.session_state: st.session_state.prompt_tokens = 0
if 'completion_tokens' not in st.session_state: st.session_state.completion_tokens = 0
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key = st.text_input("API Key (阿里云百炼)", type="password", placeholder="sk-...")
    base_url = st.text_input("Base URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    PREDEFINED_MODELS = [
        "qwen-plus", "qwen-max", "qwen-turbo", "qwen-long",
        "deepseek-chat", "deepseek-coder",
        "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
        "moonshot-v1-8k", "moonshot-v1-32k",
        "glm-4", "glm-3-turbo",
        "yi-large", "yi-medium",
        "自定义输入..."
    ]
    selected_model = st.selectbox("选择模型", PREDEFINED_MODELS)
    if selected_model == "自定义输入...":
        model_name = st.text_input("请输入自定义模型名称", value="qwen-plus")
    else:
        model_name = selected_model
        
    st.markdown("---")
    
    st.header("📊 Token 消耗监控")
    st.metric(label="累计消耗总 Token", value=f"{st.session_state.total_tokens:,}")
    col_t1, col_t2 = st.columns(2)
    with col_t1: st.metric(label="提示词", value=f"{st.session_state.prompt_tokens:,}")
    with col_t2: st.metric(label="生成词", value=f"{st.session_state.completion_tokens:,}")
    
    st.markdown("---")
    st.markdown("### 📚 挂载知识库状态 (规范RAG)")

    uploaded_file = st.file_uploader("📂 上传设计规范文件 (支持 TXT / PDF / Word)", type=["txt", "pdf", "docx", "doc"])
    
    if uploaded_file is not None and st.session_state.last_uploaded_file != uploaded_file.name:
        if not api_key:
            st.warning("⚠️ 检测到新文件上传，但尚未填写 API Key。")
        else:
            with st.spinner(f"🚀 自动构建高维向量知识库..."):
                try:
                    os.makedirs("./data", exist_ok=True)
                    file_path = os.path.join("./data", uploaded_file.name)
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())

                    embeddings = OpenAIEmbeddings(
                        api_key=api_key, base_url=base_url if base_url else None, 
                        model="text-embedding-v3", chunk_size=10, check_embedding_ctx_length=False 
                    )
                    
                    if file_path.lower().endswith(".pdf"): loader = PyPDFLoader(file_path)
                    elif file_path.lower().endswith((".docx", ".doc")): loader = Docx2txtLoader(file_path)
                    else: loader = TextLoader(file_path, encoding="utf-8")

                    docs = loader.load()
                    splits = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30).split_documents(docs)
                    
                    if not splits: st.error("❌ 文件解析为空！")
                    else:
                        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.success(f"✅ 构建成功！生成 {len(splits)} 个片段。")
                except Exception as e:
                    st.error(f"构建失败: {str(e)}")

    if os.path.exists("./chroma_db"): st.success("✅ 向量规范库 (ChromaDB) 已就绪")
    else: st.warning("⚠️ 向量库未就绪")
    
    with st.expander("查看当前本地语义映射库 (Sense Mapping)"):
        st.json(SENSE_MAPPING)

# --- 主界面：采用 Tabs 结构进行功能解耦 ---
tab1, tab2 = st.tabs(["🌳 设计需求与规范转译 (核心机制)", "🗺️ 场地现状多维解析 (新增功能)"])

# ================= TAB 1: 需求转译 =================
with tab1:
    st.markdown("将**感性自然语言描述**，智能转译为**带规范约束的参数化任务书 (JSON)**")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 输入设计需求")
        default_prompt = "设计一个社区口袋公园，里面要有一个儿童游戏场，为了美观，场地周围种一些夹竹桃。另外需要一段给老人用的轮椅小道，为了好玩坡度做到 10%。"
        user_input = st.text_area("自然语言描述：", value=default_prompt, height=200, key="req_input")
        submit_btn = st.button("🚀 执行语义解析与参数转译", use_container_width=True, type="primary")

    with col2:
        st.subheader("2. 数字化任务书 (Digital Brief)")
        if submit_btn:
            if not api_key: st.error("请先在左侧边栏填写 API Key！")
            elif not user_input.strip(): st.warning("请输入设计需求！")
            else:
                with st.spinner(f"系统正在进行意图提取与规范校验中..."):
                    try:
                        final_json, cb = generate_digital_brief(api_key, base_url, model_name, user_input)
                        
                        st.session_state.total_tokens += cb.total_tokens
                        st.session_state.prompt_tokens += cb.prompt_tokens
                        st.session_state.completion_tokens += cb.completion_tokens
                        
                        if final_json.get('warnings') and len(final_json['warnings']) > 0:
                            st.warning("⚠️ 检测到规范冲突，已自动修正：")
                            for w in final_json['warnings']: st.write(f"- {w}")
                        else:
                            st.success("✅ 规范校验通过，未发现冲突。")
                        
                        st.json(final_json)
                        
                        json_string = json.dumps(final_json, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="💾 导出数字化任务书 (.json)",
                            file_name="digital_landscape_brief.json",
                            mime="application/json",
                            data=json_string,
                            type="secondary",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"处理失败: {str(e)}")
        else:
            st.info("等待输入需求并执行...")

# ================= TAB 2: 场地解析 =================
with tab2:
    st.markdown("上传场地勘察报告文件，或输入文字描述，系统将深度挖掘并输出多维度的场地分析与设计建议。")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("1. 提供场地原始信息")
        site_file = st.file_uploader("📂 上传场地勘察报告 (支持 TXT / PDF / Word)", type=["txt", "pdf", "docx", "doc"], key="site_file")
        st.markdown("**或直接输入场地描述：**")
        default_site = "场地位于深圳市南山区某老旧社区内，面积约2000平米，三面被高层住宅环绕，日照时间较短。场地内部高差显著，最大高差约3米，目前杂草丛生，有一处常年积水的洼地。社区内老年人口占比较高，缺乏活动空间。"
        site_text = st.text_area("场地描述：", value=default_site, height=150, key="site_text")
        
        site_analyze_btn = st.button("🗺️ 开始场地多维解析", use_container_width=True, type="primary", key="site_btn")

    with col4:
        st.subheader("2. 智能多维场地分析报告")
        if site_analyze_btn:
            if not api_key:
                st.error("请先在左侧边栏填写 API Key！")
            else:
                combined_site_info = site_text
                # 如果上传了文件，临时解析文件内容合并进去
                if site_file is not None:
                    with st.spinner("正在提取上传的场地文件内容..."):
                        temp_path = os.path.join("./data", "temp_site_" + site_file.name)
                        os.makedirs("./data", exist_ok=True)
                        with open(temp_path, "wb") as f: f.write(site_file.getbuffer())
                        
                        try:
                            if temp_path.lower().endswith(".pdf"): loader = PyPDFLoader(temp_path)
                            elif temp_path.lower().endswith((".docx", ".doc")): loader = Docx2txtLoader(temp_path)
                            else: loader = TextLoader(temp_path, encoding="utf-8")
                            
                            file_docs = loader.load()
                            extracted_text = "\n".join([d.page_content for d in file_docs])
                            combined_site_info = f"【文件提取内容】\n{extracted_text}\n\n【用户附加描述】\n{site_text}"
                            os.remove(temp_path) # 用完即删
                        except Exception as e:
                            st.error(f"场地文件读取失败: {str(e)}")

                if not combined_site_info.strip():
                    st.warning("请上传场地资料文件或输入场地描述！")
                else:
                    with st.spinner(f"系统正在深度剖析场地特征..."):
                        try:
                            site_json, cb = generate_site_analysis(api_key, base_url, model_name, combined_site_info)
                            
                            st.session_state.total_tokens += cb.total_tokens
                            st.session_state.prompt_tokens += cb.prompt_tokens
                            st.session_state.completion_tokens += cb.completion_tokens
                            
                            st.success("✅ 场地解析完成！")
                            st.json(site_json)
                            
                            site_json_string = json.dumps(site_json, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="💾 导出场地分析报告 (.json)",
                                file_name="site_analysis_report.json",
                                mime="application/json",
                                data=site_json_string,
                                type="secondary",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"解析失败: {str(e)}")
        else:
            st.info("上传或输入资料后，点击按钮开始解析...")