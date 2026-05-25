# 核验版文献调研重建设计

**日期**：2026-05-26  
**输出目录**：`docs/lit_review_verified_2026-05-26/`  
**输入目录**：`docs/lit_review/`、`docs/superpowers/specs/`  
**语言**：简体中文；论文标题、会议/期刊、DOI、arXiv、BibTeX 和技术术语可保留英文。

## 1. 目标

重新围绕 `docs/lit_review/` 下的 8 个方向做文献调研，生成一个独立的新目录。旧报告只作为审计输入和主题线索，不作为可信来源。新报告中的每篇文献必须能追溯到真实来源；无法核验的旧条目不得支撑任何实质性结论。

本次工作不修改、不删除 `docs/lit_review/` 中的旧报告。

## 2. 输出范围

生成 9 个 Markdown 文件：

```text
docs/lit_review_verified_2026-05-26/
  00_synthesis_report.md
  01_safe_marl.md
  02_comm_constrained_marl.md
  03_safety_comm_intersection.md
  04_uav_formation.md
  05_multi_objective_pareto_marl.md
  06_cbf_mpc_shielding_safe_marl.md
  07_graph_attention_scalable_marl.md
  08_sim2real_uav_marl.md
```

8 个方向名称与原目录保持一致。综合报告在全部方向报告完成后生成。

## 3. 来源优先级

每条最终引用至少包含一个可追踪来源链接。优先级如下：

1. DOI、Crossref、出版社页面。
2. arXiv 官方记录。
3. OpenReview、PMLR、NeurIPS、ICML、ICLR、AAAI、AAMAS、IJCAI、CoRL、RSS、ICRA、IROS 等官方会议或论文集页面。
4. IEEE、ACM、Springer、ScienceDirect、Wiley、JMLR 等出版页面。
5. Semantic Scholar 或 OpenAlex 作为二级元数据源。

Google Scholar 只能用于发现线索，不能作为唯一核验来源。

## 4. 核验标签

原报告中的每个论文式条目使用以下标签审计：

| 标签 | 含义 | 使用规则 |
|---|---|---|
| `Verified` | 标题、作者、年份、venue 与可靠来源基本一致 | 可作为证据引用 |
| `Metadata mismatch` | 论文存在，但旧报告的作者、年份、venue、标题或发表状态有错误 | 只用修正后的元数据引用 |
| `Weak match` | 找到相似工作，但无法确认旧条目就是该论文 | 只进审计表，不用于论证 |
| `Not found` | 定向搜索后未找到可靠来源 | 只进审计表和高风险列表 |
| `Replaced` | 旧条目不适合引用，改用真实相关论文替代 | 用替代论文支撑综述 |

## 5. 单方向报告结构

每个方向报告使用统一结构：

```markdown
# 方向X：方向名称

## 1. 核验结论摘要

## 2. 原文献真实性审计表

## 3. 经核验后的核心文献综述

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

## 5. 可引用参考文献
```

`核验结论摘要` 必须包含：原报告候选条目数、Verified 数、Metadata mismatch 数、Weak match 数、Not found 数、Replaced/新增数，以及旧报告结论需要保留、降级或删除的影响。

`原文献真实性审计表` 必须包含：旧条目、状态、核验结果、可靠来源、处理方式。

`经核验后的核心文献综述` 按主题综合，而不是逐篇堆砌。所有结论只能来自 `Verified`、修正后的 `Metadata mismatch` 或新增的真实文献。

## 6. 综合报告结构

`00_synthesis_report.md` 使用以下结构：

```markdown
# 核验版文献调研综合报告

## 1. 总体核验结果
## 2. 八个方向的证据强度
## 3. 原报告需要修正或删除的结论
## 4. 仍可防守的 SafeComm 创新定位
## 5. 推荐 Related Work 写法
## 6. 高风险旧引用列表
## 7. 核验后的核心参考文献
```

研究空白表述必须保守。例如使用“本次核验未发现直接覆盖 X 的工作”，避免写成“绝对没有任何工作”。

## 7. 方法流程

1. 提取旧报告中所有论文式条目，包括章节标题、参考文献列表、BibTeX 风格条目和正文中明确引用的论文。
2. 对每个条目按“精确标题 -> 标题+作者 -> 标题+venue -> 相近主题替代”的顺序检索。
3. 记录来源链接和核验标签。
4. 剔除无法核验或弱匹配条目对结论的支撑作用。
5. 为每个方向补充真实存在且高度相关的核心文献。
6. 重写方向报告，分离“证据支持的结论”和“仍不确定的判断”。
7. 最后写综合报告，汇总证据强度和 SafeComm 论文写作建议。

## 8. 质量标准

- 新报告必须独立于旧目录生成。
- 旧目录 `docs/lit_review/` 保持只读。
- 每条最终参考文献都必须有可靠来源链接。
- 无来源或弱匹配文献不得进入参考文献列表。
- 元数据错误必须显式修正。
- 优先每个方向 10-20 篇高质量可核验文献，而不是追求大量低可信引用。
- 对 SafeComm 的设计建议必须标明证据强度：强支持、间接支持、证据不足。
- 不使用不存在的 `scientific-schematics` 技能生成图；如需要视觉化，只使用 Markdown 表格或 Mermaid 图。

## 9. 验收标准

- `docs/lit_review_verified_2026-05-26/` 存在并包含上述 9 个 Markdown 文件。
- 每个方向报告都有审计表、核验摘要、重写综述、SafeComm 关系和参考文献。
- 每个最终引用至少有一个可追踪来源链接。
- 综合报告明确列出高风险旧引用和修正后的研究定位。
- `git diff -- docs/lit_review` 无输出，证明旧报告未被修改。
