from typing import List, Union, Optional

from app.models.embeddings import EmbedResponse


class BaseModelExecutor:
    def embed(self, texts: Union[str, List[str]], prefix: Optional[str] = None) -> EmbedResponse:
        raise NotImplementedError()