from pathlib import Path
from pydantic import BaseModel
from sympy import Q


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


class Queries(BaseModel):
    query_texts: list[str]


class Config(BaseModel):
    random_seed: int
    populate_database: bool
    queries: Queries
    encoder: Encoder
    relational_database: RelationalDatabase
    vector_database: VectorDatabase
