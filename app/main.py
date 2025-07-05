from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.concurrency import run_in_threadpool
from app.models.embeddings import EmbedRequest, EmbedResponse, Embeddings
from app.services.embedding_service import EmbeddingService
from app.utils.logging import log

embedding_service = EmbeddingService()
app = FastAPI(
    title="Self-Hosted Embedding Service",
    default_response_class=ORJSONResponse
)
app = FastAPI()

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if isinstance(request.texts, list) and not request.texts:
        raise HTTPException(status_code=400, detail="Request must include at least one text.")

    log.info(f"Received embedding request. Attempting to generate embeddings...")
    try:
        embeddings: Embeddings = await run_in_threadpool(
            embedding_service.embed_text, request.texts, request.mode
        )

        log.info(f"Successfully processed embedding request.")
        return JSONResponse(
            content=EmbedResponse(embeddings=embeddings).model_dump(),
            status_code=200
        )
    except Exception as e:
        log.error(f"An error occurred while generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, e: Exception):
    log.error(f"An error occurred while generating embeddings: {e}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )