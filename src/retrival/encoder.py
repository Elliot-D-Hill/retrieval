from chromadb import (
    Documents,
    EmbeddingFunction,
    Embeddings,
)
from transformers import AutoTokenizer, AutoModel

from retrival.config import Config


class Encoder(EmbeddingFunction):
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.text_to_vector(input)
        return embeddings

    def text_to_vector(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().tolist()


def make_embedding_function(config: Config):
    # FIXME uncomment for production
    # tokenizer_path = config.encoder.huggingface / config.encoder.tokenizer
    # bcb_model_path = config.encoder.huggingface / config.encoder.bioclinicalbert
    # tokenizer = AutoTokenizer.from_pretrained(bcb_tokenizer_path)
    # model = AutoModel.from_pretrained(bcb_model_path)
    # FIXME delete for production
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return Encoder(model=model, tokenizer=tokenizer)
