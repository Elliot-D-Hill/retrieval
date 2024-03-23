import numpy as np
import json
from chromadb import PersistentClient

from retrival.config import Config


def query_vector_db(query_texts, embedding_function, config: Config):
    client = PersistentClient(path=config.vector_database.path)

    collection = client.get_collection(
        name=config.vector_database.collection_name,
        embedding_function=embedding_function,
    )
    query_results = collection.query(
        query_texts=query_texts,
        n_results=config.vector_database.k_neighbors,
        # where={"asd": 1},  # you can filter based on metadata
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
