# DS 4320 Project 2 Rubric Checklist

Converted from `project-2-rubric.pdf` and audited strictly against the current repository contents.

Legend:
- `DONE` = explicitly present in repo now.
- `NOT DONE` = missing or clearly not satisfied from repo contents.
- `UNCLEAR` = cannot be strictly verified without runtime checks or external systems (you said to flag these for manual check).

---

## General (10 pts)

### Subtotal: **5 / 10** strictly verified (+1 unclear)

- [ ] **(1 pt)** Materials submitted on time — `UNCLEAR` (manual grading event)
- [x] **(4 pts)** Well-organized GitHub repository — `DONE` (clear top-level structure and folders)
- [x] **(5 pts)** README in root, markdown, makes materials findable — `DONE` (`README.md` in root with rubric sections and links)

## Coding Standards (5 pts)

### Subtotal: **4 / 5** strictly verified (+1 unclear)

- [ ] **(1 pt)** Project written in Python, Markdown, and `mongosh` — `NOT DONE` (`.py` and `.md` present; no `mongosh` script/usage found)
- [ ] **(1 pt)** All code runs without major errors — `UNCLEAR` (requires execution)
- [x] **(1 pt)** Inline comments/docstrings explain classes/functions — `DONE` (`data/build_project2_data.py` includes function docstrings/comments)
- [x] **(1 pt)** Proper Python error handling — `DONE` (`try/except` for dataset load, DB connection, bulk writes)
- [x] **(1 pt)** Python logging to log files — `DONE` (`build_project2_data.log` and pipeline logger to `logs/project2_pipeline.log`)

## Project Details - L1 Header (10 pts)

### Subtotal: **8 / 10** strictly verified

- [x] **(1 pt)** README L1 title starts with "DS 4320 Project 2:" — `DONE`
- [x] **(3 pts)** Executive summary paragraph — `DONE`
- [x] **(1 pt)** Name — `DONE`
- [x] **(1 pt)** NetID — `DONE`
- [ ] **(1 pt)** DOI created and correctly linked — `NOT DONE` (DOI text present, but link points to Zenodo account settings page, not DOI landing URL)
- [x] **(1 pt)** Press release link — `DONE`
- [x] **(1 pt)** Pipeline link — `DONE`
- [x] **(1 pt)** License name and link to top-level file — `DONE`

## Problem Definition - L2 Header (10 pts)

### Subtotal: **10 / 10** strictly verified

- [x] **(3 pts)** General + refined specific problem statement — `DONE`
- [x] **(3 pts)** Motivation paragraph — `DONE`
- [x] **(3 pts)** Rationale for refinement paragraph — `DONE`
- [x] **(1 pt)** Press release headline + link to separate markdown file — `DONE`

## Domain Exposition - L2 Header (10 pts)

### Subtotal: **5 / 10** strictly verified

- [x] **(2 pts)** Terminology table (jargon/KPIs) — `DONE`
- [x] **(2 pts)** Domain paragraph — `DONE`
- [ ] **(1 pt)** Background reading item #1 in separate OneDrive folder — `NOT DONE` (no separate OneDrive evidence; local folder exists)
- [ ] **(1 pt)** Background reading item #2 — `NOT DONE`
- [ ] **(1 pt)** Background reading item #3 — `NOT DONE`
- [ ] **(1 pt)** Background reading item #4 — `NOT DONE`
- [ ] **(1 pt)** Background reading item #5 — `NOT DONE`
- [x] **(1 pt)** Background reading summary table with title/description/link — `DONE` (table exists in `README.md`)

## Data Creation - L2 Header (10 pts)

### Subtotal: **10 / 10** strictly verified

- [x] **(2 pts)** Provenance paragraph(s) — `DONE`
- [x] **(3 pts)** Code table (file, description, link) — `DONE`
- [x] **(3 pts)** Critical decision rationale — `DONE`
- [x] **(1 pt)** Bias identification — `DONE`
- [x] **(1 pt)** Bias mitigation — `DONE`

## Metadata - L2 Header (10 pts)

### Subtotal: **10 / 10** strictly verified

- [x] **(3 pts)** Soft-schema guidelines for document structure — `DONE`
- [x] **(1 pt)** Data summary — `DONE`
- [x] **(3 pts)** Data dictionary (name/type/description/example) — `DONE`
- [x] **(3 pts)** Data dictionary uncertainty for numeric features — `DONE`

## Press Release File (10 pts)

### Subtotal: **8 / 10** strictly verified

- [x] **(2 pts)** L1 headline — `DONE` (`press_release.md`)
- [x] **(2 pts)** Hook (L2) — `DONE`
- [x] **(2 pts)** Problem Statement (L2) — `DONE`
- [x] **(2 pts)** Solution Description (L2) — `DONE`
- [ ] **(2 pts)** Chart (L2) supporting solution — `NOT DONE` (`press_release.md` references `pipeline/roc_confusion.png`, but image file not present in repo)

## Data Stored in Mongo Atlas (10 pts)

### Subtotal: **0 / 10** strictly verified (+10 unclear)

- [ ] **(1 pt)** Dataset > 10 documents — `UNCLEAR` (requires DB check)
- [ ] **(2 pts)** Dataset > 100 documents — `UNCLEAR` (requires DB check)
- [ ] **(3 pts)** Dataset > 1000 documents — `UNCLEAR` (requires DB check)
- [ ] **(4 pts)** Canvas comment includes grader DB username/password (not on GitHub) — `UNCLEAR` (external to repo)

Note: `build_project2_data.log` suggests a successful run with 2000 inserted docs, but strict rubric verification still requires manual DB confirmation.

## Problem Solution Pipeline - Separate Files (13 pts)

### Subtotal: **13 / 13** strictly verified

- [x] **(1 pt)** Pipeline in Jupyter notebook file — `DONE` (`pipeline/project2_pipeline.ipynb`)
- [x] **(1 pt)** Notebook also saved as markdown — `DONE` (`pipeline/project2_pipeline.md`)
- [x] **(2 pts)** Data prep queries MongoDB into DataFrame — `DONE`
- [x] **(1 pt)** Analysis implements a model — `DONE` (Logistic Regression)
- [x] **(1 pt)** Analysis rationale provided — `DONE`
- [x] **(1 pt)** Uses DS 3021/4021 ML/AI complexity — `DONE`
- [x] **(1 pt)** Visualizes results — `DONE` (ROC + confusion matrix + top features in pipeline code)
- [x] **(1 pt)** Visualization rationale provided — `DONE`
- [x] **(1 pt)** Publication-quality visualization intent/settings — `DONE` (styling, DPI, labels, layout)
- [x] **(5 pts)** Pipeline solves the problem — `DONE` (binary AI-vs-human classifier end-to-end in notebook)

---

## Strict Score Snapshot

- **Strictly verified earned:** **73 / 100**
- **Not done from repo evidence:** **17 pts**
- **Unclear/manual verification needed:** **11 pts**

## Highest-Impact Fixes To Reach 100

- Add missing chart image file so `press_release.md` chart requirement is fully satisfied (+2).
- Fix DOI hyperlink to the actual DOI landing URL (+1).
- Add explicit `mongosh` usage/script evidence if required by your instructor (+1).
- Confirm Mongo Atlas scale + Canvas credential comment manually (+10).
- Clarify whether local background reading files are acceptable in place of required OneDrive folder (+5).
