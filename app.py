import streamlit as st
import os
import json
import datetime
import pandas as pd
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
# 1. 路径与数据结构定义
# ==========================================
DATA_DIR = "./data"
DB_PATH = "./chroma_db"
MAPPING_JSON_PATH = os.path.join(DATA_DIR, "sense_mapping.json")
KB_TXT_PATH = os.path.join(DATA_DIR, "规范知识库_总表.txt")
OUTPUT_DIR = "./output_briefs"

class LandscapeBrief(BaseModel):
    project_type: str = Field(description="项目类型")
    style_preference: str = Field(description="景观风格")
    parameters: dict = Field(description="物理参数键值对，如 {'郁闭度': 0.8}")
    functional_zones: list[str] = Field(description="功能分区列表")
    regulations_kv: dict = Field(description="应用的规范简要键值对")
    citation_excerpts: list[str] = Field(description="对应的国家规范原始条文摘录")
    warnings: list[str] = Field(description="修正说明或合规警告")

# ==========================================
# 2. 核心工具函数
# ==========================================

def load_mapping_kv():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if os.path.exists(MAPPING_JSON_PATH):
        with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f: return json.load(f)
    else:
        default = {"幽静": "郁闭度 > 0.7", "开阔": "郁闭度 < 0.3", "适老": "坡度 < 2.5%", "生态": "绿地率 > 70%"}
        with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f: json.dump(default, f, ensure_ascii=False)
        return default

def get_embeddings(api_key, base_url):
    """
    Embedding 专用: 大部分国产模型 Embedding 接口和 Chat 接口 Base URL 一致
    但注意：如果用通义千问，Embedding 路径也是兼容模式
    """
    return OpenAIEmbeddings(
        api_key=api_key, 
        base_url=base_url if base_url else None,
        model="text-embedding-v3", 
        chunk_size=10, 
        check_embedding_ctx_length=False
    )

def process_file_and_sync_txt(uploaded_file, api_key, base_url):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    fpath = os.path.join(DATA_DIR, uploaded_file.name)
    with open(fpath, "wb") as f: f.write(uploaded_file.getbuffer())
    
    if fpath.lower().endswith(".pdf"): loader = PyPDFLoader(fpath)
    elif fpath.lower().endswith((".docx", ".doc")): loader = Docx2txtLoader(fpath)
    else: loader = TextLoader(fpath, encoding="utf-8")
    
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    
    with open(KB_TXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*30}\n【来源文件】: {uploaded_file.name}\n【同步日期】: {datetime.datetime.now()}\n{'='*30}\n")
        f.write(full_text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    Chroma.from_documents(documents=splits, embedding=get_embeddings(api_key, base_url), persist_directory=DB_PATH)
    return len(splits)

def save_digital_report_with_citations(data):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"数字化任务书_{data.get('project_type')}_{timestamp}.txt"
    fpath = os.path.join(OUTPUT_DIR, fname)
    
    lines = [
        "==========================================",
        "      景观设计数字化任务书 (Digital Brief)",
        f"      生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "==========================================\n",
        f"【项目基本定位】\n- 项目类型: {data.get('project_type')}\n- 风格倾向: {data.get('style_preference')}\n",
        "【设计参数键值对 (Parameters KV)】"
    ]
    for k, v in data.get('parameters', {}).items(): lines.append(f" - {k}: {v}")
    lines.append("\n【合规审查建议】")
    for w in data.get('warnings', []): lines.append(f" ! {w}")
    if not data.get('warnings'): lines.append(" ✅ 意图符合规范要求。")
    
    lines.append("\n" + "-"*30 + "\n【附录：对应规范原文摘录】")
    for i, excerpt in enumerate(data.get('citation_excerpts', [])):
        lines.append(f"[{i+1}] {excerpt.strip()}")
    
    full_content = "\n".join(lines)
    with open(fpath, "w", encoding="utf-8") as f: f.write(full_content)
    return fpath, full_content

# ==========================================
# 3. Streamlit UI 界面
# ==========================================
st.set_page_config(page_title="景观需求转译全机型系统", layout="wide", page_icon="🌿")

if 'mapping_kv' not in st.session_state: st.session_state.mapping_kv = load_mapping_kv()

st.title("🌿 景观需求转译与规范同步系统 (全模型支持)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🤖 模型引擎配置")
    
    # 【已修正】各个主流厂商的标准 Base URL
    PROVIDERS = {
        "DeepSeek (深度求索)": {"url": "https://api.deepseek.com/v1", "models": ["deepseek-chat", "deepseek-coder"]},
        "阿里云百炼 (通义千问)": {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"]},
        "OpenAI (ChatGPT)": {"url": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]},
        "月之暗面 (Kimi)": {"url": "https://api.moonshot.cn/v1", "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]},
        "智谱AI (ChatGLM)": {"url": "https://open.bigmodel.cn/api/paas/v4/", "models": ["glm-4", "glm-4-flash", "glm-3-turbo"]},
        "零一万物 (Yi)": {"url": "https://api.lingyiwanwu.com/v1", "models": ["yi-large", "yi-medium", "yi-spark"]},
        "Ollama (本地私有)": {"url": "http://localhost:11434/v1", "models": ["llama3", "qwen2", "mistral"]},
        "自定义 (兼容OpenAI)": {"url": "", "models": []}
    }

    provider_name = st.selectbox("选择模型供应商", list(PROVIDERS.keys()))
    provider_cfg = PROVIDERS[provider_name]
    
    api_key = st.text_input("API Key", type="password", placeholder="填入对应平台的 API Key")
    
    # 动态 URL 控制
    base_url = st.text_input("Base URL (检查末尾是否带/v1)", value=provider_cfg["url"])
    
    # 动态模型选择
    if provider_cfg["models"]:
        model_name = st.selectbox("选择模型版本", provider_cfg["models"])
    else:
        model_name = st.text_input("手动输入模型名称", value="gpt-4o")

    st.markdown("---")
    st.header("📑 1. 语义映射管理")
    with st.expander("管理感性词映射"):
        for k, v in st.session_state.mapping_kv.items():
            st.session_state.mapping_kv[k] = st.text_input(f"{k}:", value=v, key=f"map_{k}")
        if st.button("💾 保存映射修改"): 
            with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(st.session_state.mapping_kv, f, ensure_ascii=False, indent=4)
            st.toast("映射库已更新")

    st.divider()
    st.header("📚 2. 规范同步管理")
    up_file = st.file_uploader("上传规范 (PDF/Word/TXT)", type=["pdf", "docx", "txt"])
    if up_file and api_key:
        if st.button("🛠️ 学习新规范并同步"):
            with st.spinner("解析并构建向量库中..."):
                num = process_file_and_sync_txt(up_file, api_key, base_url)
                st.success(f"同步成功！新增 {num} 片段。")
    
    if os.path.exists(KB_TXT_PATH):
        if st.button("📖 预览本地文本总库"):
            with open(KB_TXT_PATH, "r", encoding="utf-8") as f:
                st.text_area("本地文本总库预览", f.read(), height=250)

# --- 主界面 ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. 景观意图输入")
    user_desc = st.text_area("描述您的需求:", placeholder="在此输入设计意图...", height=300)
    
    if st.button("🚀 执行数字化转译", use_container_width=True, type="primary"):
        if not api_key: 
            st.error("请先在左侧输入 API Key")
        elif not base_url:
            st.error("Base URL 不能为空")
        else:
            with st.spinner(f"正在通过 {provider_name} 进行转译..."):
                try:
                    # 检索逻辑
                    context_snippets = []
                    if os.path.exists(DB_PATH):
                        vs = Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings(api_key, base_url))
                        docs = vs.as_retriever(search_kwargs={"k":4}).invoke(user_desc)
                        context_snippets = [d.page_content for d in docs]
                    
                    # LLM 转译引擎
                    llm = ChatOpenAI(
                        api_key=api_key, 
                        base_url=base_url.strip(), 
                        model=model_name, 
                        temperature=0
                    )
                    
                    parser = JsonOutputParser(pydantic_object=LandscapeBrief)
                    prompt = PromptTemplate(
                        template="""你是一位景观合规专家与任务书架构师。
                        【需求】: {u}
                        【语义库】: {m}
                        【规范条文】: {r}
                        要求：
                        1. 转译为 parameters KV。
                        2. 审查是否冲突。
                        3. 从【规范条文】中摘录核心原句存入 citation_excerpts。
                        4. 修正冲突并在 warnings 记录。
                        格式: {f}""",
                        input_variables=["u", "m", "r"],
                        partial_variables={"f": parser.get_format_instructions()}
                    )
                    
                    res = (prompt | llm | parser).invoke({
                        "u": user_desc,
                        "m": json.dumps(st.session_state.mapping_kv, ensure_ascii=False),
                        "r": "\n".join(context_snippets) if context_snippets else "库中暂无相关条文"
                    })
                    
                    fpath, fcontent = save_digital_report_with_citations(res)
                    st.session_state.final_res = (res, fcontent, fpath)
                except Exception as e:
                    st.error(f"⚠️ 转译失败！\n\n**错误代码**: {str(e)}")
                    if "404" in str(e):
                        st.info("💡 **排查建议**: 检测到 404 错误。这通常是 Base URL 路径不对。对于 DeepSeek 请确保地址以 `/v1` 结尾。")

with col_right:
    st.subheader("2. 数字化任务书预览")
    if 'final_res' in st.session_state:
        res, txt, path = st.session_state.final_res
        st.markdown("#### 核心参数 KV")
        st.json(res.get('parameters'))
        
        if res.get('warnings'):
            for w in res['warnings']: st.warning(f"修正建议: {w}")
        else:
            st.success("✅ 符合国家强制性规范。")
        
        with st.expander("🔍 查看引用规范原文"):
            for i, cite in enumerate(res.get('citation_excerpts', [])):
                st.caption(f"[{i+1}] {cite}")
        
        st.text_area("数字化报告预览:", txt, height=300)
        st.download_button("💾 下载报告", txt, file_name=os.path.basename(path))
