from enum import StrEnum
from typing import List, Union, Literal, Optional

from pydantic import BaseModel

InputTexts = Union[str, List[str]]
Embeddings = Union[List[float], List[List[float]]]

class EmbeddingsMode(StrEnum):
    QUERY = "query"
    PASSAGE = "passage"

class EmbedRequest(BaseModel):
    texts: InputTexts
    mode: Optional[EmbeddingsMode] = "query"

class EmbedResponse(BaseModel):
    embeddings: Embeddings