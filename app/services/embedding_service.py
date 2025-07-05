import os
from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from app.models.embeddings import Embeddings


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes mean pooling while ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class EmbeddingService:
    """
    Functional and optimized embedding service using HuggingFace models.
    Supports automatic token-chunking with overlap to avoid max-length errors.
    """

    def __init__(self):
        model_name = os.environ.get("MODEL_NAME", "intfloat/multilingual-e5-small")
        detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._device = os.environ.get("DEVICE", detected_device)
        self._model.to(self._device)
        self._max_length = int(os.environ.get("MAX_CHUNK_LENGTH", 512))
        self._stride = int(os.environ.get("STRIDE", 50))
        self._embed_prefix = os.environ.get("EMBED_PREFIX", "query: ")
        self._normalize = os.environ.get("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
        self._padding = os.environ.get("PADDING_MODE", "max_length")

    def embed_text(
            self, texts: Union[str, List[str]], embed_prefix: Optional[str] = None
    ) -> Embeddings:
        """
        Embed a single string or a list of strings.
        Returns a list of floats for a single input, or a list of lists for multiple inputs.
        """
        single = isinstance(texts, str)
        texts_list: List[str] = [texts] if single else texts  # type: ignore

        outputs: List[List[float]] = []
        for text in texts_list:
            prefix = (embed_prefix or self._embed_prefix).lower()
            full_text = f"{prefix}{text.strip()}"

            # Tokenize with automatic chunking and overlap
            encodings = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                stride=self._stride,
                return_overflowing_tokens=True,
                padding=self._padding,
            ).to(self._device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            # Forward pass once for all chunks
            with torch.no_grad():
                model_outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            # Pool and normalize each chunk
            chunk_embeddings = _mean_pooling(
                model_outputs.last_hidden_state, attention_mask
            )
            if self._normalize:
                chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

            # Average chunk embeddings into a single vector
            avg_embedding = chunk_embeddings.mean(dim=0).cpu().tolist()
            outputs.append(avg_embedding)

        return outputs[0] if single else outputs
