from chromadb import PersistentClient
from toml import load
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from transformers import logging
import json

from retrival.config import Config
from retrival.database import populate_database
from retrival.encoder import make_embedding_function


# FIXME delete for production
def make_fake_database(config: Config):
    data = {
        "mrn": [1, 2, 3, 4, 5],
        "asd": [1, 1, 0, 0, 1],
        "age": [0.1, 0.5, 1.0, 2.0, 3.0],
        "text": [
            "The quick red fox jumps over the lazy dog.",
            "A quick brown fox leaped over the lazy dog.",
            "An apple a day keeps the doctor faraway.",
            "You don't have to see doctors often if you eat an apple every day",
            "The architecture of the building is modern.",
        ],
        "note_id": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    engine = create_engine(config.relational_database.filepath)
    with engine.connect() as conn:
        df.to_sql(name="clinical_notes", con=conn, if_exists="replace")


def main():
    # TODO delete me; only used to silence huggingface warnings
    logging.set_verbosity_error()
    config_data = load("config.toml")
    config = Config(**config_data)
    # TODO delete for production
    make_fake_database(config=config)
    query_texts = [
        "A fast black cat leaped over the sleepy wolf",
        "An apple pie a week keeps the physician away",
    ]
    print(query_texts)
    embedding_function = make_embedding_function(config=config)
    client = PersistentClient(path=config.vector_database.path)
    collection = client.get_or_create_collection(
        name=config.vector_database.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    if config.populate_database:
        populate_database(collection=collection, config=config)
    # you can filter on the metadata using the 'where' arg
    # For example, where={"asd": 1, "age" {"$lte": 2.0}}
    # will return only documents where asd is 1 and age is less than or equal to 2.0
    query_results = collection.query(
        query_texts=query_texts,
        n_results=config.vector_database.k_neighbors,
        where=None,  # TODO add filtering
        include=[
            "documents",
            "distances",
            "metadatas",
            "embeddings",
        ],
    )

    print(query_results.keys())
    print("ids:", query_results["ids"])
    print("nearest neighbors:", query_results["documents"])
    print("neighbor distances:", query_results["distances"])
    embeddings = np.array(query_results["embeddings"])
    print(
        "embeddings shape (n_queries, k_neighbors, embedding dimension):",
        embeddings.shape,
    )
    print("metadata:", json.dumps(query_results["metadatas"], indent=4))


if __name__ == "__main__":
    main()
