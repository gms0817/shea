from typing import List, Union, Optional

from pydantic import BaseModel

InputTexts = Union[str, List[str]]
Embeddings = Union[List[float], List[List[float]]]

class EmbedRequest(BaseModel):
    texts: InputTexts
    embed_prefix: Optional[str]

class EmbedResponse(BaseModel):
    embeddings: Embeddings