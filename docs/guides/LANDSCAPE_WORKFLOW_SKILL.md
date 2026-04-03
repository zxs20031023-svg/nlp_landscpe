---
name: landscape-workflow
description: Maintain and extend the Landscape NLP workflow project in this repository. Use when working on the Streamlit app, knowledge-base ingestion, compliance rules, RAG retrieval, landscape design brief generation, site analysis, or related workflow changes.
---

# Landscape Workflow

Use this file as the project skill prompt when continuing development on this repository.

## Work Sequence

1. Read `需求文档.md` before changing behavior.
2. Keep the Streamlit UI shell in root `app.py` and the real page logic in `src/landscape_workflow/app.py`.
3. Keep knowledge-base logic in `src/landscape_workflow/knowledge_base.py`.
4. Keep prompt logic in `src/landscape_workflow/llm.py`.
5. Keep deterministic compliance logic in `src/landscape_workflow/rules.py`.
6. Keep orchestration in `src/landscape_workflow/service.py`.
7. Prefer editing `config/sense_mapping.json` and `config/compliance_rules.json` before hardcoding new thresholds.

## Output Principles

- Keep the core deliverable JSON-first.
- Put warnings, retrieval basis, and history around the main result rather than mixing extra prose into the JSON payload.
- Preserve `runtime/output_briefs/` as the runtime artifact directory and store workflow history files with the `history_` prefix.

## Extension Guide

- Add new compliance thresholds to `config/compliance_rules.json`.
- Add new post-processing logic to `src/landscape_workflow/rules.py`.
- Add new orchestration steps to `src/landscape_workflow/service.py`.
- Add new UI affordances to root `app.py` or `src/landscape_workflow/app.py` as appropriate.
