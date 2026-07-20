"""
Handles all interaction with the local ChromaDB vector store:
  - Creating/loading a persistent client
  - Populating it once from data/sports_facts.json
  - Querying it for facts relevant to a chosen sport
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DB_PATH, SPORTS_FACTS_PATH

COLLECTION_NAME = "sports_history"


def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client saved to disk."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_collection(client):
    """Gets (or creates) the sports_history collection with a local embedding function."""
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )


def setup_and_populate_db():
    """
    Reads the offline JSON facts, creates the collection, and populates it.
    Safe to call on every app start-up: it skips re-inserting if data already exists.
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    if collection.count() > 0:
        print(f"[database] Already populated with {collection.count()} facts.")
        return collection

    if not os.path.exists(SPORTS_FACTS_PATH):
        print(f"[database] Fact file not found at {SPORTS_FACTS_PATH}")
        return collection

    with open(SPORTS_FACTS_PATH, "r") as f:
        facts_list = json.load(f)

    documents, metadata_list, ids = [], [], []
    for idx, item in enumerate(facts_list):
        documents.append(item["fact"])
        metadata_list.append({"sport": item["sport"]})
        ids.append(f"fact_{idx}")

    collection.add(documents=documents, metadatas=metadata_list, ids=ids)
    print(f"[database] Vectorized and stored {len(documents)} facts.")
    return collection


def query_historic_facts(sport: str, query_text: str, n_results: int = 3):
    """
    Queries ChromaDB for documents relevant to `query_text`, filtered to `sport`.
    Returns a list of matched fact strings (possibly empty).
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        where={"sport": sport},
    )
    return results.get("documents", [[]])[0]
