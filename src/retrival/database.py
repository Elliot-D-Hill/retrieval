from retrival.config import Config
from sqlalchemy import create_engine, text
from chromadb import Metadata, PersistentClient

from retrival.encoder import make_embedding_function, Encoder


def create_chunks(text: str, chunk_length: int, overlap: int):
    return [
        text[start : start + chunk_length]
        for start in range(0, len(text), chunk_length - overlap)
    ]


def populate_database(embedding_function: Encoder, config: Config):
    client = PersistentClient(path=config.vector_database.path)
    embedding_function = make_embedding_function(config=config)
    collection = client.get_or_create_collection(  # FIXME create_collection
        name=config.vector_database.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    engine = create_engine(config.relational_database.filepath)
    with engine.connect() as conn:
        query = text(f"SELECT * FROM {config.relational_database.table_name}")
        result = conn.execute(query)
        i = 0
        while True:
            row = result.mappings().fetchone()
            if row is None:
                break
            row = dict(row)
            note_text = row["text"]  # .pop("text")  # FIXME "Note_text"
            chunks = create_chunks(
                note_text,
                chunk_length=config.vector_database.chunk_length,
                overlap=config.vector_database.overlap,
            )
            n_chunks = len(chunks)
            next_i = i + n_chunks
            ids = [str(id_i) for id_i in range(i, next_i)]
            i = next_i
            metadatas: list[Metadata] = [row] * n_chunks
            collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )
