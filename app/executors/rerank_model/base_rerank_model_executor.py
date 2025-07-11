
from typing import List, Union, Optional

from app.models.reranking import RerankResponse


class BaseRerankModelExecutor:
    def rerank(self, query: str, retrieved_texts: List[str]) -> RerankResponse:
        raise NotImplementedError()