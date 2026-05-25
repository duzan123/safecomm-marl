# Verified Literature Review Redesign

## Purpose

Rebuild the existing SafeComm-MARL literature review set under a new directory without modifying the original files in `docs/lit_review/`. The current reports contain citations that are not consistently traceable to DOI, arXiv, publisher pages, OpenReview, Semantic Scholar, or OpenAlex. The new reports must audit the original citations, remove or downgrade unverifiable claims, add verified core literature, and rewrite the conclusions on a defensible evidence base.

## Output Scope

Create a new directory:

```text
docs/lit_review_verified/
```

Generate nine Markdown documents:

```text
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

The original directory `docs/lit_review/` is read-only for this task.

## Language Requirement

All newly generated literature review reports under `docs/lit_review_verified/` must be written in Simplified Chinese. Paper titles, venue names, technical terms, DOI strings, arXiv identifiers, BibTeX fields, and citation metadata may remain in English when that improves traceability.

## Method

Use the domain-reinforced rewrite approach:

1. Extract all paper-like entries from the eight direction reports.
2. Verify each original entry against traceable sources.
3. Mark original entries as `Verified`, `Metadata mismatch`, `Weak match`, `Not found`, or `Replaced`.
4. Replace unverifiable or weak entries with real, relevant, traceable papers.
5. Rewrite each direction report around verified evidence, not around the original narrative.
6. Rewrite the synthesis report to reflect only evidence that survived verification or is explicitly marked as uncertain.

## Source Priority

Use at least two source categories where feasible. Source priority:

1. DOI and Crossref metadata.
2. arXiv records.
3. OpenReview, conference proceedings, or official conference pages.
4. Publisher and proceedings pages, including IEEE, ACM, Springer, ScienceDirect, PMLR, AAAI, IJCAI, AAMAS, RSS, CoRL, ICRA, IROS, NeurIPS, ICML, ICLR.
5. Semantic Scholar or OpenAlex as secondary metadata and citation graph sources.

Google Scholar may be used only as a discovery aid, not as the sole verification source.

## Per-Direction Report Template

Each of the eight direction reports will use this structure:

```markdown
# Direction X: Title

## 1. Verification Summary
- Original citation count
- Verified count
- Metadata mismatch count
- Weak match count
- Not found count
- Replacement or newly added citation count
- Impact on the original report's conclusions

## 2. Audit Table for Original Citations
| Original entry | Status | Verification result | Reliable source | Treatment |
|---|---|---|---|---|

## 3. Verified Literature Review
Thematic synthesis based only on verified or explicitly qualified literature.

## 4. Relevance to SafeComm-MARL
What the verified literature supports, what it does not support, and which original claims should be retained, downgraded, or removed.

## 5. Verified References
Each reference includes author, title, venue, year, and at least one traceable source link.
```

## Synthesis Report Template

The rewritten `00_synthesis_report.md` will use this structure:

```markdown
# Verified Literature Review Synthesis

## 1. Overall Citation Verification
## 2. Evidence Strength Across Eight Directions
## 3. Original Conclusions Requiring Revision
## 4. SafeComm-MARL Innovation Claims That Remain Defensible
## 5. Recommended Related Work Framing
## 6. High-Risk Citation List
## 7. Verified Core References
```

## Verification Labels

- `Verified`: title, authors, year, and venue substantially match a reliable source.
- `Metadata mismatch`: the paper exists, but the original report has incorrect authors, year, venue, title, or publication status.
- `Weak match`: only a similar paper was found; the original entry cannot be confidently verified.
- `Not found`: no reliable source was found after targeted searches.
- `Replaced`: the original entry is unsuitable for citation and a verified related paper is used instead.

## Quality Standards

- Do not edit files under `docs/lit_review/`.
- Every cited paper in the new reports must include at least one traceable source.
- Unverified entries cannot support substantive claims.
- Metadata corrections must explicitly state the corrected title, authors, year, and venue when available.
- Prefer 10-20 high-quality verified papers per direction over larger but weaker citation lists.
- Claims about research gaps must be written conservatively: absence of found evidence is not proof that no work exists.
- The synthesis report must separate verified conclusions from uncertain or evidence-limited observations.

## Figure Constraint

The literature-review skill recommends adding generated scientific schematics. The required `scientific-schematics` skill is not available in this environment. The implementation will prioritize citation verification and textual rigor. If a visual aid is useful, use Markdown tables or Mermaid diagrams instead of AI-generated figures.

## Acceptance Criteria

- `docs/lit_review_verified/` exists and contains exactly the nine requested Markdown reports.
- Original reports in `docs/lit_review/` remain unchanged.
- Each direction report contains an original-citation audit table and a rewritten verified review.
- Each new report marks unverifiable or mismatched original citations clearly.
- Every retained or newly added reference has a traceable source link.
- The synthesis report updates SafeComm-MARL's positioning based on verified evidence only.
- The final response lists the created files and notes any verification limitations.
