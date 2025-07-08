import os
import platform
from collections.abc import Callable
from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from app.models.embeddings import Embeddings

from app.utils.logging import log


class EmbeddingService:
    """
    Functional and optimized embedding service using HuggingFace models.
    Supports automatic token-chunking with overlap to avoid max-length errors.
    """

    def __init__(self):
        model_name = os.environ.get("MODEL_NAME")
        detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_torch()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        use_8bit = bool(os.environ.get("USE_8BIT", False))
        if use_8bit:
            torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self._device = os.environ.get("DEVICE", detected_device)
        self._model.to(self._device)
        self._truncation = bool(os.environ.get("USE_TRUNCATION", True))
        self._max_length = int(os.environ.get("MAX_CHUNK_LENGTH", 512))
        self._stride = int(os.environ.get("STRIDE", 50))
        self._embed_prefix = os.environ.get("EMBED_PREFIX", None)
        self._normalize = bool(os.environ.get("NORMALIZE_EMBEDDINGS", True))
        self._padding = os.environ.get("PADDING_MODE", "max_length")
        pooling_mode = os.environ.get("POOLING_MODE", "auto")  # auto | mean | last_token
        self._pooling_fn: Callable = self._get_pooling_fn(
            pooling_mode=pooling_mode,
            model_name=model_name,
        )

    def embed_text(
            self, texts: Union[str, List[str]], embed_prefix: Optional[str] = None
    ) -> Embeddings:
        """
        Embed a single string or a list of strings.
        Returns a list of floats for a single input, or a list of lists for multiple inputs.
        """
        single = isinstance(texts, str)
        texts: List[str] = [texts] if single else texts
        outputs: List[List[float]] = []
        prefix: Optional[str] = embed_prefix.lower() if embed_prefix else None
        if prefix:
            texts = [f"{prefix}{text.strip()}" for text in texts]

        # Tokenize with automatic chunking and overlap
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

        # Forward pass once for all chunks
        with torch.no_grad():
            model_outputs = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # Pool and normalize each chunk
        chunk_embeddings = self._pooling_fn(
            model_outputs.last_hidden_state, attention_mask
        )
        if self._normalize:
            chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

        # Average chunk embeddings into a single vector
        avg_embedding = chunk_embeddings.mean(dim=0).cpu().tolist()
        outputs.append(avg_embedding)

        return outputs[0] if single else outputs

    def _last_token_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), sequence_lengths]

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _prepare_torch(self):
        num_threads = int(os.environ.get("NUM_THREADS", 2))
        gradient_tracking = bool(os.environ.get("USE_GRADIENT_TRACKING", False))
        use_8bit: bool = bool(os.environ.get("USE_8BIT", False))

        torch.set_num_threads(num_threads)
        torch.set_grad_enabled(gradient_tracking)
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
