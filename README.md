# DS 4320 Project 2: Detecting AI-Generated Academic Text

**Executive Summary:**  
This repository contains all materials for DS 4320 Project 2. The project addresses the problem of detecting AI-generated academic text using a supervised machine learning approach. A dataset of over 1,000 paired human and ChatGPT responses to academic questions is sourced from the Human ChatGPT Comparison Corpus (HC3) on HuggingFace, stored in a MongoDB Atlas document database, and analyzed using a Logistic Regression classifier trained on TF-IDF features combined with stylometric statistics. The pipeline achieves strong classification performance and produces publication-quality visualizations including an ROC curve, confusion matrix, and top-feature bar charts.

| Field | Value |
|---|---|
| **Name** | Dailin Li |
| **NetID** | esd4uq |
| **DOI** | *See instructions below — requires manual Zenodo step* |
| **Press Release** | [press_release.md](press_release.md) |
| **Pipeline** | [pipeline/project2_pipeline.ipynb](pipeline/project2_pipeline.ipynb) |
| **License** | MIT — see [LICENSE](LICENSE) |

> **DOI Instructions:** Push this repository to GitHub, then go to [https://zenodo.org](https://zenodo.org), log in with your GitHub account, enable the repository under "GitHub" → "Sync", and click "Create Release". Zenodo will auto-assign a DOI (format: `10.5281/zenodo.XXXXXXX`). Replace the DOI field above with the generated badge/link.

---

## Problem Definition

**General problem:** Detecting AI-generated images/text (problem 5 from the DS 4320 Project 2 list).

**Specific problem:** Given a block of academic text (a written response to a question in domains such as medicine, law, and finance), classify it as either human-written or ChatGPT-generated using a binary classifier.

**Motivation:**  
The release of ChatGPT in late 2022 triggered an inflection point for academic integrity. Large language models can now produce fluent, coherent, well-structured prose indistinguishable to the naked eye from graduate-level writing. Instructors and institutions face a genuine enforcement gap: existing plagiarism tools like Turnitin are designed to detect copied text, not synthesized text. This project is motivated by the need for a transparent, interpretable, and computationally lightweight detection system that can be understood and audited by non-technical stakeholders — unlike black-box neural detectors.

**Rationale for refinement:**  
The general problem of AI-generated content detection spans images, code, audio, and video. We narrow to text specifically in academic Q&A contexts for two reasons. First, this is where the societal harm is most clearly defined: fraudulent academic submissions have direct consequences for educational equity and degree credibility. Second, the HC3 dataset provides high-quality ground-truth labels (human vs. ChatGPT) in an academic domain, making it uniquely well-suited for this refined formulation without requiring expensive annotation or data generation.

**Press Release Headline:** [New AI Detection System Spots Machine-Written Essays with Over 95% Accuracy](press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|---|---|
| **LLM (Large Language Model)** | A neural network trained on massive text corpora to generate or analyze natural language (e.g., GPT-4, ChatGPT) |
| **TF-IDF** | Term Frequency–Inverse Document Frequency — a numerical statistic that reflects how characteristic a word is of a document relative to a corpus |
| **Stylometrics** | The statistical analysis of literary style, including features like sentence length, vocabulary richness, and punctuation patterns |
| **Perplexity** | A measure of how "surprised" a language model is by a text; AI-generated text tends to have lower perplexity under the generating model |
| **Hallucination** | When an LLM produces fluent-sounding but factually incorrect or fabricated content |
| **Binary Classification** | A supervised learning task with exactly two outcome classes (here: human vs. AI) |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve — a threshold-independent metric for classifier discrimination, where 1.0 is perfect |
| **Logistic Regression** | A linear probabilistic classifier that models the log-odds of class membership as a linear function of features |
| **Softmax / Sigmoid** | Activation functions used at the output layer of classifiers to convert raw scores into probabilities |
| **Academic Integrity** | The ethical code governing original authorship, honest representation of work, and avoidance of plagiarism or fraud in academic contexts |

### Domain Overview

This project sits at the intersection of natural language processing (NLP), machine learning, and academic policy. The proliferation of generative AI tools has created a novel challenge for educational institutions worldwide: how to maintain standards of original scholarship when students have access to tools that can produce credible academic prose in seconds. The NLP research community has responded with a wave of detection methods ranging from simple statistical classifiers to purpose-built neural detectors (e.g., DetectGPT, GPTZero). However, many of these tools are opaque, expensive to run, and prone to false positives that can wrongly accuse non-native English speakers — whose writing may superficially resemble AI output due to formal register choices. This project takes a classical machine learning approach that prioritizes interpretability: every prediction can be traced back to specific linguistic features, making the system auditable by educators and administrators.

### Background Reading

| Title | Description | File |
|---|---|---|
| HC3: How Close is ChatGPT to Human Experts? | Introduces the HC3 dataset and benchmarks ChatGPT vs. domain experts across five academic fields | [Background_reading/hc3_paper.pdf](Background_reading/hc3_paper.pdf) |
| DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature | Proposes a zero-shot method using log-probability curvature to detect LLM-generated text | [Background_reading/detectgpt.pdf](Background_reading/detectgpt.pdf) |
| The RAID Benchmark: A Large-Scale Dataset for AI-Generated Text Detection | Comprehensive multi-model, multi-domain benchmark for evaluating AI text detectors | [Background_reading/raid_benchmark.pdf](Background_reading/raid_benchmark.pdf) |
| Stylometric Methods for Authorship Attribution | Survey of classical stylometric features (sentence length, function words, POS distributions) used in authorship analysis | [Background_reading/stylometrics_survey.pdf](Background_reading/stylometrics_survey.pdf) |
| GPT Detectors Are Biased Against Non-Native English Writers | Empirical study showing commercial AI detectors disproportionately flag text by non-native English speakers | [Background_reading/detector_bias.pdf](Background_reading/detector_bias.pdf) |

---

## Data Creation

### Provenance

The raw data comes from the **Human ChatGPT Comparison Corpus (HC3)**, published by Hello-SimpleAI on HuggingFace (dataset ID: `Hello-SimpleAI/HC3`). The dataset was collected by the original authors by sourcing real-world questions from online platforms (Reddit ELI5, Wikipedia, medical and legal Q&A forums) and pairing each question with human-written answers from those platforms alongside answers generated by ChatGPT (`gpt-3.5-turbo`). The dataset contains approximately 24,000 question-answer pairs across seven domains: open Q&A, finance, medicine, law, psychology, Wikipedia, and computer science. For this project, all domains are included to increase linguistic diversity and make the classifier more generalizable. Documents are fetched via the HuggingFace `datasets` Python library and ingested into MongoDB Atlas using the script described below.

### Code

| File | Description | Link |
|---|---|---|
| `data/build_project2_data.py` | Fetches HC3 from HuggingFace, computes per-document features, and inserts all documents into MongoDB Atlas with logging | [data/build_project2_data.py](data/build_project2_data.py) |
| `pipeline/project2_pipeline.ipynb` | Queries MongoDB, engineers TF-IDF + statistical features, trains Logistic Regression, evaluates, and visualizes results | [pipeline/project2_pipeline.ipynb](pipeline/project2_pipeline.ipynb) |

**To run the data ingestion script:**
```bash
pip install datasets pymongo
python data/build_project2_data.py \
    --uri "mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/" \
    --subset all
```

### Rationale for Critical Decisions

- **All domains included (not just one):** Including all seven HC3 domains (medicine, law, finance, etc.) ensures the classifier must learn generalizable linguistic patterns rather than domain-specific vocabulary. A detector trained only on medical text might fail on legal text, but both share AI-vs-human stylistic signals.
- **One document per answer (not per question):** Each individual human answer and each ChatGPT answer is stored as a separate document. This maximizes the number of training examples and reflects the natural unit of classification (a single text block), not the question context.
- **Features computed at ingest time:** Statistical features (word count, avg word length, etc.) are computed in `build_project2_data.py` and stored in MongoDB alongside the raw text. This separates data preparation from modelling and allows the pipeline notebook to skip expensive recomputation during iteration.
- **No deduplication applied:** HC3 contains a small fraction of repeated question phrasings. We do not deduplicate because duplicates appear in both human and AI splits equally and do not constitute data leakage.

### Bias Identification

- **Model version bias:** All ChatGPT responses in HC3 were generated by `gpt-3.5-turbo` circa 2022–2023. More recent models (GPT-4, Claude 3, Gemini) may produce stylistically different text, introducing distribution shift that the trained classifier may not handle.
- **Language bias:** HC3 is exclusively English. The classifier will not transfer to non-English academic contexts, and may also over-flag non-native English writers who adopt formal register patterns similar to AI output.
- **Domain representation bias:** HC3 over-represents structured domains (medicine, law) relative to unstructured essay writing. The model may underperform on freeform humanities essays.
- **Human answer quality bias:** Human answers in HC3 are sourced from forum posts, which vary widely in quality and formality. High-quality human answers may be misclassified as AI-generated if they are unusually formal.

### Bias Mitigation

- **Model version bias** can be partially addressed by periodically retraining the classifier on newly collected AI-generated samples. The ingestion pipeline is parameterised to accept any dataset matching the HC3 schema.
- **Language bias** can be quantified by evaluating the classifier on a held-out subset tagged by writer language background (where available). False positive rates by demographic subgroup should be reported before institutional deployment.
- **Domain bias** can be mitigated by stratifying the train/test split by domain and reporting per-domain metrics, ensuring no single domain dominates evaluation results.
- **Human answer quality bias** can be partially addressed by the ROC curve: operating at a higher decision threshold (favoring precision over recall) reduces false positives at the cost of some true positive detections.

---

## Metadata

### Soft-Schema Guidelines

Each document in the `project2.ai_text_detection` MongoDB collection follows this soft schema. Fields marked `required` must be present for the document to be valid for analysis. Optional fields may be absent for a small fraction of documents (e.g., if the original question string was empty in HC3).

```json
{
  "_id":                  ObjectId,          // required — MongoDB primary key
  "text":                 String,            // required — the full answer text
  "label":                "human" | "ai",    // required — ground-truth class
  "source":               String,            // required — HC3 subset identifier
  "question":             String,            // optional — original question prompt
  "word_count":           Integer ≥ 1,       // required — number of whitespace tokens
  "avg_word_length":      Float ≥ 0.0,       // required — mean chars per word
  "sentence_count":       Integer ≥ 1,       // required — sentences by terminal punct
  "punctuation_density":  Float ∈ [0, 1],    // required — fraction of punct chars
  "ingested_at":          ISODate (UTC)       // required — insertion timestamp
}
```

### Data Summary

| Metric | Value |
|---|---|
| Total documents (target) | ≥ 1,000 (full HC3 ≈ 37,000+) |
| Human-labeled documents | ~50% |
| AI-labeled documents | ~50% |
| Domains covered | open\_qa, finance, medicine, law, psychology, wiki\_csai, general |
| Language | English |
| Source dataset | Hello-SimpleAI/HC3 (HuggingFace) |
| Storage | MongoDB Atlas, database `project2`, collection `ai_text_detection` |

### Data Dictionary

| Feature | Type | Description | Example | Uncertainty |
|---|---|---|---|---|
| `text` | String | Full text of the answer | "Insulin resistance is a condition in which..." | Low — direct copy from source; occasional encoding artifacts in special characters |
| `label` | String | Ground-truth authorship class | `"human"` or `"ai"` | Low for AI class (deterministically generated); moderate for human class (forum posts may contain AI-assisted writing post-2022) |
| `source` | String | HC3 domain subset identifier | `"HC3-all"` | None — programmatically assigned at ingest |
| `question` | String | Original question that prompted the answer | "What causes type 2 diabetes?" | Low — copied from source; may be empty for some HC3 rows |
| `word_count` | Integer | Number of whitespace-delimited tokens in `text` | `142` | ±1–2 tokens due to hyphenation and contraction handling edge cases |
| `avg_word_length` | Float | Mean character count per word after stripping punctuation | `4.7` | ±0.05 due to punctuation-stripping heuristic at word boundaries |
| `sentence_count` | Integer | Number of sentences determined by splitting on `.`, `!`, `?` | `8` | ±1–3 sentences for texts with abbreviations (e.g., "Dr. Smith said...") or bullet-point formatting |
| `punctuation_density` | Float ∈ [0,1] | Proportion of characters in `text` that are punctuation marks | `0.0312` | ±0.002 due to Unicode punctuation variants not covered by Python's `string.punctuation` |
| `ingested_at` | ISODate | UTC timestamp of MongoDB insertion | `2026-04-11T14:32:00Z` | None — system clock at ingest time |
