import platform

from transformers import AutoTokenizer, AutoModel
from app.executors.base_model_executor import BaseModelExecutor
from app.models.embeddings import EmbedResponse
import torch
import torch.nn.functional as F
import os
from typing import List, Union, Callable

from app.utils.logging import log


class HuggingFaceModelExecutor(BaseModelExecutor):
    def __init__(self, model_name: str, device: str):
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        use_8bit = bool(os.environ.get("USE_8BIT", False))
        if use_8bit:
            torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self._max_length = int(os.environ.get("MAX_CHUNK_LENGTH", 512))
        self._stride = int(os.environ.get("STRIDE", 50))
        self._truncation = bool(os.environ.get("USE_TRUNCATION", True))
        self._normalize = bool(os.environ.get("NORMALIZE_EMBEDDINGS", True))
        self._padding = os.environ.get("PADDING_MODE", "max_length")
        pooling_mode = os.environ.get("POOLING_MODE", "auto")  # auto | mean | last_token
        self._pooling_fn: Callable = self._get_pooling_fn(
            pooling_mode=pooling_mode,
            model_name=model_name,
        )
        self._prepare_torch()
        self._model.to(self._device)

    def embed(self, texts: Union[str, List[str]], prefix: str = None) -> EmbedResponse:
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        if prefix:
            texts = [f"{prefix}{t.strip()}" for t in texts]

        encodings = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self._max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            padding=self._padding,
        ).to(self._device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._pooling_fn(outputs.last_hidden_state, attention_mask)
        if self._normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        if pooled.shape[0] > 1:
            avg = pooled.mean(dim=0).cpu().tolist()
        else:
            avg = pooled.squeeze(0).cpu().tolist()

        log.debug(f"Embedded {len(texts)} text(s) using HF model on {self._device}")
        return EmbedResponse(dense=avg if single else [avg])

    def _get_pooling_fn(self, *, pooling_mode: str, model_name: str) -> Callable:
        match pooling_mode:
            case "mean":
                return self._mean_pool
            case "last_token":
                return self._last_token_pool
            case "auto":
                model_lower = model_name.lower()
                if "qwen" in model_lower or "gpt" in model_lower:
                    return self._last_token_pool
                elif "bge" in model_lower or "e5" in model_lower or "jina" in model_lower:
                    return self._mean_pool
                else:
                    log.warn(f"Unknown model '{model_name}', defaulting to mean pooling.")
                    return self._mean_pool
            case _:
                log.warn(f"Unknown pooling mode '{pooling_mode}', defaulting to mean pooling.")
                return self._mean_pool


    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _last_token_pool(self, last_hidden_state, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]

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
