from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from logger import get_logger


sep_map = {
    'en': ' ',
    'zh': '',
    'fr': ' ',
    'de': ' ',
    'cs': ' ',
    'es': ' ',
    'it': ' ',
    'pt': ' ',
    'ru': ' ',
    'ja': '',
    'ar': ' ',
    'English': ' ',
    'Chinese': '',
    'French': ' ',
    'German': ' ',
    'Czech': ' ',
    'Spanish': ' ',
    'Italian': ' ',
    'Portuguese': ' ',
    'Russian': ' ',
    'Japanese': '',
    'Arabic': ' '
}

logger = get_logger(__name__)

class SentencePairs:
    def __init__(self, encoder: SentenceTransformer, src_sentences: str = None, tgt_sentences: str = None, src_embeddings=None, src_lang: str = 'en'):
        self.encoder = encoder
        self.src_sentences = src_sentences if src_sentences is not None else []
        self.tgt_sentences = tgt_sentences if tgt_sentences is not None else []
        self.src_lang = src_lang
        if src_embeddings is not None:
            if isinstance(src_embeddings, list):
                self.src_embeddings = np.array(src_embeddings)
            elif isinstance(src_embeddings, np.ndarray):
                self.src_embeddings = src_embeddings
            else:
                raise ValueError("src_embeddings should be a list or numpy array.")
        elif self.src_sentences != []:
            self.src_embeddings = self.encoder.encode(self.src_sentences)
        else:
            self.src_embeddings = None

    def to_dict(self):
        return {
            "src_sentences": self.src_sentences,
            "tgt_sentences": self.tgt_sentences,
            "src_embeddings": self.src_embeddings.tolist() if self.src_embeddings is not None else None,
            "src_lang": self.src_lang,
        }
    
    @classmethod
    def from_dict(cls, data, encoder: SentenceTransformer):
        sentence_pairs = cls(
            encoder=encoder,
            src_sentences=data["src_sentences"],
            tgt_sentences=data["tgt_sentences"],
            src_embeddings=data["src_embeddings"] if data["src_embeddings"] is not None else None,
            src_lang=data.get("src_lang", 'en')
        )
        return sentence_pairs

    def update_pairs(self, src_sentences: list[str], tgt_sentences: list[str]):
        logger.debug("SentencePairs update_pairs called.")
        new_embeddings = self.encoder.encode(src_sentences)
        print(f"src_sentences length: {len(src_sentences)}, tgt_sentences length: {len(tgt_sentences)}, src_embeddings shape: {new_embeddings.shape}")
        assert len(src_sentences) == len(tgt_sentences) == (new_embeddings.shape[0]), f"Length of src_sentences, tgt_sentences and src_embeddings must match: {len(src_sentences)}, {len(tgt_sentences)}, {new_embeddings.shape[0]}"
        self.src_sentences.extend(src_sentences)
        self.tgt_sentences.extend(tgt_sentences)
        if self.src_embeddings is None:
            self.src_embeddings = new_embeddings
        else:
            self.src_embeddings = np.vstack((self.src_embeddings, new_embeddings))

    def get_similarities(self, query_sentences: list[str]):
        assert (self.src_sentences == [] and self.src_embeddings is None) or len(self.src_sentences) == self.src_embeddings.shape[0], "Length of src_sentences and src_embeddings must match."
        if self.src_embeddings is None:
            raise ValueError("No source embeddings available. Please add source sentences first.")
        query_embedding = self.encoder.encode(sep_map[self.src_lang].join(query_sentences))
        similarities = self.encoder.similarity(query_embedding, self.src_embeddings)
        return similarities
    
    def get_exemplars(self, query_sentences: list[str], top_k: int = 4):
        logger.debug(f"SentencePairs get_exemplars called. top_k={top_k}")
        similarities = self.get_similarities(query_sentences)
        top_values, top_indices = torch.topk(similarities, k=min(top_k, similarities.shape[1]))
        top_indices = top_indices.tolist()[0]
        logger.debug(f"Top-{top_k} ids: {top_indices}")
        print(f"Top-{top_k} ids: {top_indices}")
        candidate_pairs = [(self.src_sentences[i], self.tgt_sentences[i]) for i in top_indices]
        return candidate_pairs
