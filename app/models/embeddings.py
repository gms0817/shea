from typing import List, Union, Optional, Dict

from pydantic import BaseModel

InputTexts = Union[str, List[str]]
Vectors = Union[List, List[Dict[str, float]], List[List]]

class EmbedRequest(BaseModel):
    texts: InputTexts
    embed_prefix: Optional[str] = None

class EmbedResponse(BaseModel):
    dense: Optional[Vectors] = None
    sparse: Optional[Vectors] = None
    colbert: Optional[Vectors] = None

