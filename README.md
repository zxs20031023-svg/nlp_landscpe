# 景观设计需求转译与场地分析工作流

这是一个基于 `Streamlit + LangChain + OpenAI Compatible API + ChromaDB` 的景观设计前期分析项目，面向课程展示、设计研究和轻量业务辅助场景。项目支持规范知识库构建、场地分析、数字化任务书生成、相似项目推荐，以及在线案例检索与入库。

## 当前版本能力

- 规范文件上传、解析、切分、去重与 Chroma 向量库持久化
- 基于规范检索的 RAG 任务书生成
- 显式规则校验与风险修正提示
- 场地多维分析与结构化 JSON 输出
- 本地相似案例推荐
- 在线案例检索、勾选导入、本地案例库沉淀
- 数字化任务书固定目录归档
- 项目历史记录与结果资产查看

## 项目结构

```text
app.py
src/
  landscape_workflow/
    app.py
    config.py
    knowledge_base.py
    llm.py
    loaders.py
    models.py
    persistence.py
    project_recommender.py
    rules.py
    service.py
config/
  compliance_rules.json
  knowledge_aliases.json
  sense_mapping.json
resources/
  knowledge_base/
  reference_projects/
  samples/
docs/
  guides/
  product/
  requirements/
  research/
runtime/
  chroma_db/
  output_briefs/
tests/
  test_recommendations.py
  test_rules.py
```

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后在左侧填写：

- `API Key`
- `Base URL`
- `主模型`
- `嵌入模型`

默认 `Base URL` 为阿里云百炼兼容地址，可按实际服务商调整。

## 推荐使用流程

1. 在“知识库管理”中导入内置规范或上传规范文件。
2. 在“场地分析”中输入场地描述，生成结构化分析结果。
3. 查看系统自动推荐的相似项目，并勾选需要同步到任务书的参考案例。
4. 在“设计任务书”中输入需求，生成数字化任务书。
5. 在“项目历史”中查看归档记录。

## 重要输出目录

- 规范向量库：`runtime/chroma_db/`
- 历史记录：`runtime/output_briefs/`
- 数字化任务书：`runtime/output_briefs/digital_briefs/`
- 本地案例库：`resources/reference_projects/project_case_library.json`

## 最新版本更新

- 修复了界面中原始 HTML 片段被直接显示的问题
- 调整了顶部布局，避免导航栏与内容区重叠
- 新增数字化任务书固定归档能力
- 新增本地案例推荐能力
- 新增在线案例检索、勾选导入、本地沉淀能力
- 新增参考案例同步到任务书成果文件的能力

## 测试

```bash
python -m unittest tests.test_rules tests.test_recommendations -v
```

## 文档入口

- [使用说明](D:\Landscape_NLP_Project\docs\guides\使用说明.md)
- [需求文档](D:\Landscape_NLP_Project\docs\requirements\需求文档.md)
- [产品分析与产品文档](D:\Landscape_NLP_Project\docs\product\产品分析与产品文档.md)
