import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.executors.rerank_model.base_rerank_model_executor import BaseRerankModelExecutor
from app.models.reranking import RerankResponse


class HuggingFaceRerankModelExecutor(BaseRerankModelExecutor):
    def __init__(self, model_name: str, device: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._truncation = bool(os.environ.get("RERANKER_USE_TRUNCATION", True))
        self._max_length = int(os.environ.get("RERANKER_MAX_CHUNK_LENGTH", 512))
        self._padding = os.environ.get("RERANKER_PADDING_MODE", "longest")
        self._model.to(device)

    def rerank(self, query: str, retrieved_texts: List[str]) -> RerankResponse:
        pairs: List[tuple[str, str]] = [(query, text) for text in retrieved_texts]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=self._padding,
                truncation=self._truncation,
                return_tensors='pt',
                max_length=self._max_length
            )
            scores: List[float] = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
            reranked_texts: List[str] = [text for _, text in sorted(zip(scores, retrieved_texts), reverse=True)]
            return RerankResponse(ranked_results=reranked_texts)
