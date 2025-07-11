# A (S)elf-(H)osted (E)mbedding (S)ervice - Shea

## What is Shea?
Shea is a Self-Hosted Embedding Service compatible with embedding models hosted on HuggingFace.
Shea enables affordable, private embedding generation for sensitive use cases where data ownership and residency are critical.

---

## Features
- Privacy First: No on‑disk storage—embeddings are generated in memory and discarded after each request.
- Multi‑lingual: Supports text in numerous languages via the multilingual-e5-small model.
- Easy Deployment: Single Docker container—no additional dependencies required.
- Health Checks: Built‑in endpoint for readiness and liveness probes.
- Configurable: Expose ports and bind addresses to suit your network policies.

## Prerequisites
- Docker ≥ 20.10
- (Optional) A firewall or reverse proxy (e.g., NGINX) for secure access

## Disclaimer
- At the time of preparing this repo, I was new to embedding models and am partially using this repo as a learning opportunity while building out a service to use for embedding sensitive data. If anything is inaccurate or can be improved upon, please let me know! Thank you!

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/gms0817/shea.git
   cd shea
   ```

2. **Build the Docker image**

   ```bash
   docker build -t shea .
   ```

3. **Run the container**

   ```bash
   docker run -d \
     --name shea \
     --restart unless-stopped \
     -p 127.0.0.1:8000:80 \
     shea
   ```

4. **Test the API**

   ```bash
   curl -X POST http://127.0.0.1:8000/embed \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Hello world", "Bonjour le monde"]}'
   ```

   **Response:**

    ```json
    {
        "dense": [
            [0.01, -0.12, ...],
            [0.11,  0.05, ...]
        ],
        "sparse": [  // If enabled
            {
                "indices": [12, 53, 99],
                "values": [0.8, 0.4, 0.1]
            }
        ],
        "colbert": [  // If enabled
            [
                [0.01, 0.02, ...],  // token 1
                [0.03, 0.04, ...]   // token 2
            ],
            [
                [0.05, 0.06, ...],
                [0.07, 0.08, ...]
            ]
        ]
    }
    ```

## Configuration
| Env Variable            | Default                          | Description                                                                                   |
|-------------------------|----------------------------------|-----------------------------------------------------------------------------------------------|
| `MODEL_NAME`            | `intfloat/multilingual-e5-small` | Hugging Face model identifier                                                                 |
| `DEVICE`                | *(auto-detected)*                | Force device selection: `cpu` or `cuda`. Leave unset to auto-detect.                         |
| `EMBED_PREFIX`          | `query:`                         | Prefix prepended to text (e.g., `query:`, `passage:`). Leave empty for none.                 |
| `NORMALIZE_EMBEDDINGS`  | `true`                           | Whether to L2-normalize embeddings (`true`/`false`).                                          |
| `MAX_CHUNK_LENGTH`      | `512`                            | Max tokens per chunk (triggers chunking on longer texts).                                    |
| `STRIDE`                | `50`                             | Number of overlapping tokens between chunks.                                                 |
| `PADDING_MODE`          | `max_length`                     | Tokenizer padding strategy: `max_length` (pad to fixed length) or `longest` (pad to longest).|
| `CONCURRENCY_LIMIT`     | `10`                             | Max concurrent requests handled by Uvicorn.                                                  |
| `PORT`                  | `80`                             | Port FastAPI server listens on inside the container.                                         |
| `LOG_LEVEL`             | `INFO`                           | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.                      |
| `POOLING_MODE`          | `auto`                           | Pooling strategy: `mean`, `last_token`, or `auto` (detects based on model type).             |
| `NUM_THREADS`           | `2`                              | Number of CPU threads to allocate for PyTorch inference.                                     |
| `USE_GRADIENT_TRACKING` | `False`                          | Whether to enable gradient tracking (typically `False` for inference-only models).           |
| `USE_TRUNCATION`        | `True`                           | Whether to truncate input text to `MAX_CHUNK_LENGTH`.                                        |
| `USE_8BIT`              | `False`                          | Whether to load the model in 8-bit precision (e.g., with `bitsandbytes` or dynamic quantization). |
| `EMBEDDING_BATCH_SIZE`  | `4`                              | Number of inputs processed per batch. Increase for better throughput (depends on RAM/CPU).   |
| `RETURN_SPARSE`         | `False`                          | Whether to return sparse lexical weights from the BGE-M3 model.                              |
| `RETURN_COLBERT`        | `False`                          | Whether to return ColBERT-style multi-vector embeddings.                                     |
| `USE_FP16`              | `False`                          | Whether to load the model in FP16 precision (if hardware supports it).                       |

You can override these by passing `-e` flags in your `docker run` command, e.g.:

```bash
docker run -d -e MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" -p 8000:80 shea
````

## API Reference

### `GET /healthz`

* **Description**: Health check endpoint.
* **Response**: `200 OK` when the service is ready. Returns an empty body or `{}` depending on configuration.

### `POST /embed`

* **Description**: Generate embeddings for one or more texts.

* **Request Body** (application/json):

  ```json
  {
    "texts": ["text1", "text2", ...],
    "embed_prefix": "passage: "
  }
  ```

   * `texts` (required): array of strings to embed. Use a single-element array for one text input.
   * `embed_prefix` (optional): The prefix to prepend to the provided texts. Defaults to `"query: "` for compatibility with the default model: multilingual-e5-small.

* **Responses**:

   * `200 OK`

     ```json
     {
       "embeddings": [
         [0.01, -0.12, ...],
         [0.11,  0.05, ...]
       ]
     }
     ```
   * `422 Unprocessable Entity` — validation error

     ```json
     {
       "detail": [
         {
           "loc": ["body", "texts"],
           "msg": "field required",
           "type": "value_error.missing"
         }
       ]
     }
     ```


## Security

* **Network**: Expose only on internal interfaces or behind a VPN/firewall.
* **TLS**: Terminate SSL in a reverse proxy (e.g., NGINX, Traefik).
* **Authentication**: Integrate with API gateways or token proxies if needed.

## FAQs

**Q: What ports should I open?**
* A: Only the host port you bind (e.g., `8000`) needs to be reachable by clients.

**Q: Does Shea store embeddings or data?**
* A: No. Embeddings are generated in memory per request and not persisted.

**Q: How do I secure the service?**
* A: Use a reverse proxy with HTTPS, restrict access via firewall or allow‑listed IPs, and/or integrate with an API gateway.
