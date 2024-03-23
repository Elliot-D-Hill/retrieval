from pathlib import Path
from pydantic import BaseModel


class Encoder(BaseModel):
    huggingface: Path
    bioclinical_bert: Path
    tokenizer: Path


class RelationalDatabase(BaseModel):
    filepath: str
    table_name: str


class VectorDatabase(BaseModel):
    path: str
    collection_name: str
    table_name: str
    chunk_length: int
    overlap: int
    k_neighbors: int


class Config(BaseModel):
    random_seed: int
    populate_database: bool
    encoder: Encoder
    relational_database: RelationalDatabase
    vector_database: VectorDatabase
