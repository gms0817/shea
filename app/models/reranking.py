from typing import List

from pydantic import BaseModel

class RerankRequest(BaseModel):
    query: str
    retrieved_texts: List[str]

class RerankResponse(BaseModel):
    ranked_results: List[str]