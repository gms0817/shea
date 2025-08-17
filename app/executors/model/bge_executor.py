import os
from typing import Union, List, Optional, Any, Dict

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from app.executors.model.base_model_executor import BaseModelExecutor
from app.models.embeddings import EmbedResponse


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0 or not np.isfinite(norm):
        return vec  # leave as-is; caller may handle
    return (arr / norm).tolist()


class BGEModelExecutor(BaseModelExecutor):
    def __init__(self, model_name: str):
        use_fp16 = getenv_bool("USE_FP16", False)

        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16
        )
        self._batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", 4))
        self._max_length = int(os.environ.get("MAX_CHUNK_LENGTH", 8192))
        self._return_sparse = getenv_bool("RETURN_SPARSE", False)
        self._return_colbert = getenv_bool("RETURN_COLBERT", False)
        self._normalize_dense = getenv_bool("NORMALIZE_EMBEDDINGS", False)

    def embed(self, texts: Union[str, List[str]], prefix: Optional[str] = None) -> EmbedResponse:
        # Note: Prefixes are not required or used in BGE-M3; kept for interface compatibility
        if isinstance(texts, str):
            texts = [texts]

        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert
        )

        return self._convert_to_serializable_response(output)

    def _to_sparse(self, lw: Any) -> Optional[Dict[str, List]]:
        indices: List[int]
        values: List[float]

        if hasattr(lw, "indices") and hasattr(lw, "data"):
            idx = getattr(lw, "indices")
            val = getattr(lw, "data")
            indices = [int(i) for i in (idx.tolist() if hasattr(idx, "tolist") else list(idx))]
            values = [float(v) for v in (val.tolist() if hasattr(val, "tolist") else list(val))]

        elif isinstance(lw, dict):
            pairs = []
            for k, v in lw.items():
                try:
                    i = int(k)
                    f = float(v)
                except (ValueError, TypeError):
                    continue
                if f == 0.0 or not np.isfinite(f):
                    continue
                pairs.append((i, f))
            if not pairs:
                return None
            dedup: Dict[int, float] = {}
            for i, f in pairs:
                dedup[i] = f
            items = sorted(dedup.items(), key=lambda x: x[0])
            indices = [int(i) for i, _ in items]
            values = [float(f) for _, f in items]
        else:
            return None

        if not indices or not values:
            return None
        for a, b in zip(indices, indices[1:]):
            if not (a < b):
                zipped = sorted({i: v for i, v in zip(indices, values)}.items())
                indices = [i for i, _ in zipped]
                values = [v for _, v in zipped]
                break

        return {"indices": indices, "values": values}

    def _convert_to_serializable_response(self, output: dict) -> EmbedResponse:
        # Dense
        dense_vecs_raw = output.get("dense_vecs", [])
        dense_vecs: List[List[float]] = []
        for v in dense_vecs_raw:
            vec = v.tolist() if hasattr(v, "tolist") else list(v)
            if self._normalize_dense:
                vec = l2_normalize(vec)
            dense_vecs.append(vec)
        dense: Union[List[float], List[List[float]]] = (
            dense_vecs[0] if len(dense_vecs) == 1 else dense_vecs
        )

        # ColBERT vectors (optional)
        colbert_vecs = output.get("colbert_vecs", []) if self._return_colbert else []
        colbert = [v.tolist() for v in colbert_vecs] if self._return_colbert else None

        # Sparse (optional)
        sparse: Optional[List[Optional[Dict[str, List]]]] = None
        if self._return_sparse:
            lw_list = output.get("lexical_weights", [])
            sparse = [self._to_sparse(lw) for lw in lw_list]

        return EmbedResponse(
            dense=dense,
            sparse=sparse,
            colbert=colbert
        )

    def _convert_to_list_safe(self, obj: Any):
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return [obj]
