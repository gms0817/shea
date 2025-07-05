from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from app.models.embeddings import Embeddings, EmbeddingsMode


class EmbeddingService:
    def __init__(self):
        huggingface_model: str = "intfloat/multilingual-e5-small"
        self._tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self._model = AutoModel.from_pretrained(huggingface_model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._chunk_word_threshold: int = 350
        self._chunk_size: int = 400
        self._overlap: int = 50

    def embed_text(self, texts: Union[str, List[str]], mode: Optional[EmbeddingsMode]) -> Embeddings:
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]

        all_embeddings = []
        for text in texts:
            formatted = f"{mode or "query"}: {text.strip()}"
            tokens = self._tokenizer.encode(formatted, add_special_tokens=False)
            if len(tokens) > self._chunk_size:
                chunks = self._chunk_by_tokens(formatted)
                chunk_embeds = [self._embed_single(chunk) for chunk in chunks]
                avg_embed = torch.tensor(chunk_embeds).mean(dim=0).tolist()
                all_embeddings.append(avg_embed)
            else:
                embed = self._embed_single(formatted)
                all_embeddings.append(embed)


        return all_embeddings[0] if is_single_input else all_embeddings


    def _embed_single(self, text: str) -> List[float]:
        encoded = self._tokenizer(
            text,
            padding=True,
            max_length=self._chunk_size,
            return_tensors='pt'
        ).to(self._device)

        with torch.no_grad():
            model_output = self._model(**encoded)

        embedding = self._mean_pooling(model_output.last_hidden_state, encoded['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().tolist()

    def _chunk_by_tokens(self, text: str):
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self._chunk_size)
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self._chunk_size - self._overlap
        return chunks

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        masked_hidden = last_hidden_state * input_mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_hidden / torch.clamp(sum_mask, min=1e-9)