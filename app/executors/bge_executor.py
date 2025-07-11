import os
from typing import Union, List, Optional

from app.executors.base_model_executor import BaseModelExecutor
from FlagEmbedding import BGEM3FlagModel

from app.models.embeddings import EmbedResponse


class BGEModelExecutor(BaseModelExecutor):
    def __init__(self, model_name: str):
        use_fp_16 = bool(os.environ.get("USE_FP16", False))
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp_16
        )
        self._batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", 4))
        self._max_length = int(os.environ.get("MAX_CHUNK_LENGTH", 8192))
        self._return_sparse = bool(os.environ.get("RETURN_SPARSE", False))
        self._return_colbert = bool(os.environ.get("RETURN_COLBERT", False))

    def embed(self, texts: Union[str, List[str]], prefix: Optional[str] = None) -> EmbedResponse:
        # Note: Prefixes are not required or used in BGE-M3
        # The `prefix` arg is retained for BaseModelExecutor interface compatibility

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

        return self._convert_to_serializable_response(
            output=output,
        )

    def _convert_to_serializable_response(self, output: dict) -> EmbedResponse:
        dense_vecs = [v.tolist() for v in output["dense_vecs"]]
        colbert_vecs = output.get("colbert_vecs", [])
        dense = dense_vecs[0] if len(dense_vecs) == 1 else dense_vecs
        colbert = [v.tolist() for v in colbert_vecs] if self._return_colbert else None
        return EmbedResponse(
            dense=dense,
            sparse=[
                {"indices": w["indices"], "values": w["values"]}
                for w in output.get("lexical_weights", [])
            ] if self._return_sparse else None,
            colbert=colbert
        )

