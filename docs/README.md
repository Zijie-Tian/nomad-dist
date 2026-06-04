# docs/ — NoMAD-Attention 文档库

本目录收纳围绕 **NoMAD-Attention**（arXiv:2403.01273, NeurIPS 2024）的深度调研文档，按文档在研究流程中的角色分类管理。原文档存放于 `outputs/`，现整理至此。

## 目录结构

| 目录 | 用途 |
|------|------|
| `reports/` | 最终交付报告 |
| `drafts/` | 中间草稿（合成稿、带引用稿） |
| `research-notes/` | 一手研究笔记 |
| `planning/` | 研究计划与任务清单 |
| `provenance/` | 来源溯源与事实核查日志 |

## 文档清单

### reports/ — 最终报告
- `nomad-attention-cpu-inference.md` — NoMAD-Attention 深度解析最终报告（问题背景、三大核心技术、码本学习、实验结果、采用现状与局限，含完整 Sources）

### drafts/ — 草稿
- `nomad-attention-cpu-inference-draft.md` — 合成草稿（无行内引用标注）
- `nomad-attention-cpu-inference-cited.md` — 带行内引用的草稿（最终报告的直接前身）

### research-notes/ — 研究笔记
- `nomad-attention-cpu-inference-research-direct.md` — 一手研究笔记：搜索词、主要来源、算法机制、码本学习、性能数据、局限

### planning/ — 计划
- `nomad-attention-cpu-inference.md` — 研究计划：主题、关键问题、证据需求、规模决策、任务清单、决策日志

### provenance/ — 溯源与核查
- `nomad-attention-cpu-inference-provenance.md` — 来源清单（10 个来源）、核查结论、研究文件索引
- `nomad-attention-cpu-inference-verification.md` — 事实核查日志：逐条核对论文定量/机制/分发/采用声明

## 研究流程脉络

```
planning → research-notes → drafts (draft → cited) → provenance (verification) → reports
```

> 这套文档源自 2026-05-15 的一轮 deep-research（单主题、lead-owned、无 subagent）。
> 相关：复现实验代码见 `experiments/`，复现日志见 `experiments/nomad-reproduction-log.md`。
