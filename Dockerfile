FROM python:3.13-slim

# Install the OS-Level dependencies for Transformers/PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*


# Create non-root user for security
RUN useradd -m serviceuser


# Define the working directory for the container
WORKDIR /code


# Install Python Dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app


# Switch to the non-root user
USER serviceuser

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers", "--limit-concurrency", "10"]
