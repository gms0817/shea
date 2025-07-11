
import os
import platform

import torch
from typing import List, Union, Optional

from app.executors.model.base_model_executor import BaseModelExecutor
from app.executors.model.bge_executor import BGEModelExecutor
from app.executors.model.huggingface_model_executor import HuggingFaceModelExecutor

from app.executors.rerank_model.base_rerank_model_executor import BaseRerankModelExecutor
from app.executors.rerank_model.huggingface_rerank_model_executor import HuggingFaceRerankModelExecutor
from app.models.embeddings import EmbedResponse
from app.models.reranking import RerankResponse


class EmbeddingService:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
        rerank_model_name = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

        detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected_device = os.getenv("DEVICE", detected_device)

        if "bge-m3" in model_name:
            self.adapter: BaseModelExecutor = BGEModelExecutor(model_name)
        else:
            self.adapter: BaseModelExecutor = HuggingFaceModelExecutor(model_name, selected_device)

        self.rerank_adapter: BaseRerankModelExecutor = HuggingFaceRerankModelExecutor(rerank_model_name, selected_device)
        self.embed_prefix = os.getenv("EMBED_PREFIX", None)
        self._prepare_torch()

    def embed_text(self, texts: Union[str, List[str]], prefix: Optional[str] = None) -> EmbedResponse:
        if isinstance(self.adapter, BGEModelExecutor):
            return self.adapter.embed(texts)
        else:
            return self.adapter.embed(texts, prefix=prefix or self.embed_prefix)

    def rerank(self, query: str, retrieved_texts: List[str]) -> RerankResponse:
        return self.rerank_adapter.rerank(query, retrieved_texts)

    def _prepare_torch(self):
        torch.set_num_threads(int(os.environ.get("NUM_THREADS", 2)))
        torch.set_grad_enabled(bool(os.environ.get("USE_GRADIENT_TRACKING", False)))
        use_8bit: bool = bool(os.environ.get("USE_8BIT", False))
        if use_8bit:
            self._set_quant_backend()

    def _set_quant_backend(self):
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin":  # macOS
            torch.backends.quantized.engine = "qnnpack"
        elif system == "windows":
            torch.backends.quantized.engine = "fbgemm"
        elif system == "linux":
            # Use qnnpack on ARM CPUs, fbgemm on x86
            if "arm" in machine or "aarch64" in machine:
                torch.backends.quantized.engine = "qnnpack"
            else:
                torch.backends.quantized.engine = "fbgemm"
        else:
            raise RuntimeError(f"Unsupported platform for quantization: {system}")
