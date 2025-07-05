# A (S)elf-(H)osted (E)mbedding (S)ervice - Shea

## What is Shea?
Shea is a Self-Hosted Embedding Service that uses the [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) embedding model for efficient multi-lingual embedding generation. Shea enables affordable, private embedding generation for sensitive use cases where data ownership and residency are critical.

---

## How to Deploy Shea with Docker

### 1. Build the Docker image
Clone this project on your server, then from the project root:
```bash
docker build -t shea .
```

### 2. Run the container
```
docker run -d --name shea --restart unless-stopped -p 127.0.0.1:8000:80 shea
```

## FAQs
What ports should I open?

    The host port you expose with -p (e.g., 8000) must be open to the clients you want to access the API.

Does Shea store embeddings or data?

    No. Embeddings are generated in-memory per request and not stored.

How do I secure it?

    Use a reverse proxy (e.g., NGINX) with HTTPS, or restrict access to the Docker-exposed port using a firewall or allowlisted IPs.
