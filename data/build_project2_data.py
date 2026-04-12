"""
build_project2_data.py

Fetches the Hello-SimpleAI/HC3 dataset (academic split) from HuggingFace,
computes lightweight text features for each sample, and inserts all documents
into a MongoDB Atlas collection.

Usage:
    python build_project2_data.py --uri <MONGO_URI> [--db <db>] [--collection <col>] [--limit <n>]
"""

import argparse
import logging
import re
import string
import sys
from datetime import datetime, timezone

from datasets import load_dataset
from pymongo import MongoClient, errors

# ── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("build_project2_data.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── text feature helpers ─────────────────────────────────────────────────────

def word_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def avg_word_length(text: str) -> float:
    """Mean character length of words, excluding punctuation tokens."""
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 4)


def sentence_count(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def punctuation_density(text: str) -> float:
    """Fraction of characters that are punctuation marks."""
    if not text:
        return 0.0
    punct_chars = sum(1 for c in text if c in string.punctuation)
    return round(punct_chars / len(text), 4)


def build_document(question: str, answer: str, label: str, source: str) -> dict:
    """Assemble a MongoDB document from one text sample."""
    return {
        "text": answer,
        "label": label,
        "source": source,
        "question": question,
        "word_count": word_count(answer),
        "avg_word_length": avg_word_length(answer),
        "sentence_count": sentence_count(answer),
        "punctuation_density": punctuation_density(answer),
        "ingested_at": datetime.now(timezone.utc),
    }


# ── data loading ─────────────────────────────────────────────────────────────

def load_hc3(subset: str = "all", limit: int | None = None) -> list[dict]:
    """
    Load HC3 from HuggingFace and convert to a flat list of documents.
    Each row in HC3 has one question, a list of human answers, and a list of
    ChatGPT answers.  We emit one document per individual answer.
    """
    logger.info("Loading HC3 dataset (subset=%s) from HuggingFace ...", subset)
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", subset)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        raise

    docs = []
    split = ds["train"]  # HC3 ships as a single 'train' split
    logger.info("Raw HC3 rows: %d", len(split))

    for row in split:
        question = row.get("question", "")
        for answer in row.get("human_answers", []):
            if answer and answer.strip():
                docs.append(build_document(question, answer, "human", f"HC3-{subset}"))
        for answer in row.get("chatgpt_answers", []):
            if answer and answer.strip():
                docs.append(build_document(question, answer, "ai", f"HC3-{subset}"))

    logger.info("Documents extracted: %d", len(docs))

    if limit:
        # keep a balanced sample: half human, half ai
        human_docs = [d for d in docs if d["label"] == "human"][:limit // 2]
        ai_docs = [d for d in docs if d["label"] == "ai"][:limit // 2]
        docs = human_docs + ai_docs
        logger.info("After balanced limit (%d): %d documents", limit, len(docs))

    return docs


# ── MongoDB insertion ─────────────────────────────────────────────────────────

def insert_documents(uri: str, db_name: str, collection_name: str, docs: list[dict]) -> int:
    """
    Connect to MongoDB Atlas and bulk-insert documents.
    Returns the number of successfully inserted documents.
    """
    logger.info("Connecting to MongoDB Atlas ...")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        client.admin.command("ping")
        logger.info("Connected successfully.")
    except errors.ServerSelectionTimeoutError as exc:
        logger.error("Could not reach MongoDB: %s", exc)
        raise

    db = client[db_name]
    collection = db[collection_name]

    # drop existing data so re-runs are idempotent
    existing = collection.count_documents({})
    if existing > 0:
        logger.warning("Collection already has %d docs — dropping before re-insert.", existing)
        collection.drop()

    logger.info("Inserting %d documents into %s.%s ...", len(docs), db_name, collection_name)
    try:
        BATCH = 200
        inserted = 0
        for i in range(0, len(docs), BATCH):
            result = collection.insert_many(docs[i:i+BATCH], ordered=False)
            inserted += len(result.inserted_ids)
            logger.info("Inserted %d documents.", inserted)
        return inserted
    except errors.BulkWriteError as exc:
        logger.error("Bulk write error: %s", exc.details)
        raise


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project 2 MongoDB dataset from HC3.")
    parser.add_argument("--uri", required=True, help="MongoDB Atlas connection URI")
    parser.add_argument("--db", default="project2", help="Database name (default: project2)")
    parser.add_argument("--collection", default="ai_text_detection", help="Collection name")
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "open_qa", "finance", "medicine", "law", "psychology", "wiki_csai"],
        help="HC3 subset to load (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max documents to insert (balanced human/ai). Default: no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== build_project2_data.py started ===")
    logger.info("DB=%s  Collection=%s  Subset=%s  Limit=%s", args.db, args.collection, args.subset, args.limit)

    docs = load_hc3(subset=args.subset, limit=args.limit)

    human_n = sum(1 for d in docs if d["label"] == "human")
    ai_n = sum(1 for d in docs if d["label"] == "ai")
    logger.info("Label breakdown — human: %d  ai: %d", human_n, ai_n)

    inserted = insert_documents(args.uri, args.db, args.collection, docs)
    logger.info("=== Done. Total inserted: %d ===", inserted)


if __name__ == "__main__":
    main()
