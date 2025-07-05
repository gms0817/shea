from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from app.models.embeddings import Embeddings, EmbeddingsMode


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

    def __init__(
            self,
            model_name: str = "intfloat/multilingual-e5-small",
            device: Optional[torch.device] = None,
            max_length: int = 512,
            stride: int = 50,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_length = max_length
        self._stride = stride

    def embed_text(
            self, texts: Union[str, List[str]], mode: Optional[EmbeddingsMode] = None
    ) -> Embeddings:
        """
        Embed a single string or a list of strings.
        Returns a list of floats for a single input, or a list of lists for multiple inputs.
        """
        single = isinstance(texts, str)
        texts_list: List[str] = [texts] if single else texts  # type: ignore

        outputs: List[List[float]] = []
        for text in texts_list:
            prefix = (mode or "query").lower()
            full_text = f"{prefix}: {text.strip()}"

            # Tokenize with automatic chunking and overlap
            encodings = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                stride=self._stride,
                return_overflowing_tokens=True,
                padding="max_length",
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
            chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

            # Average chunk embeddings into a single vector
            avg_embedding = chunk_embeddings.mean(dim=0).cpu().tolist()
            outputs.append(avg_embedding)

        return outputs[0] if single else outputs
