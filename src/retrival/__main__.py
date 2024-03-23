from toml import load
import pandas as pd
from retrival.database import populate_database
from retrival.encoder import make_embedding_function
from retrival.retriver import query_vector_db
from sqlalchemy import create_engine
from transformers import logging

from retrival.config import Config


def main():
    logging.set_verbosity_error()  # FIXME delete me; only used to silence huggingface warnings
    config_data = load("config.toml")
    config = Config(**config_data)

    # create fake sqlite database
    data = {
        "mrn": [1, 2, 3, 4, 5],
        "asd": [1, 1, 0, 0, 1],
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
        # FIXME change if_exists="append"
        df.to_sql(name="clinical_notes", con=conn, if_exists="replace")

    query_texts = [
        "A fast black cat leaped over the sleepy wolf",
        "An apple turnover a week keeps the physician away",
    ]
    print(query_texts)
    embedding_function = make_embedding_function(config=config)
    if config.populate_database:
        populate_database(embedding_function=embedding_function, config=config)
    query_vector_db(
        query_texts=query_texts, embedding_function=embedding_function, config=config
    )


if __name__ == "__main__":
    main()
