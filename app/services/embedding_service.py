
import os
import torch
from typing import List, Union, Optional

from app.executors.base_model_executor import BaseModelExecutor
from app.executors.bge_executor import BGEModelExecutor
from app.executors.huggingface_model_executor import HuggingFaceModelExecutor
from app.models.embeddings import EmbedResponse

class EmbeddingService:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
        detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected_device = os.getenv("DEVICE", detected_device)

        if "bge-m3" in model_name:
            self.adapter: BaseModelExecutor = BGEModelExecutor(model_name)
        else:
            self.adapter: BaseModelExecutor = HuggingFaceModelExecutor(model_name, selected_device)

        self.embed_prefix = os.getenv("EMBED_PREFIX", None)

    def embed_text(self, texts: Union[str, List[str]], prefix: Optional[str] = None) -> EmbedResponse:
        if isinstance(self.adapter, BGEModelExecutor):
            return self.adapter.embed(texts)
        else:
            return self.adapter.embed(texts, prefix=prefix or self.embed_prefix)
