FROM python:3.13-slim

# Install OS-level dependencies for Transformers/PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m serviceuser

# Define the working directory
WORKDIR /code

# Install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy app code
COPY ./app /code/app

# Copy start script
COPY ./start.sh /start.sh
RUN chmod +x /start.sh

# Switch to non-root user
USER serviceuser

# Default port
EXPOSE 80

# Run with custom entrypoint script
CMD ["/start.sh"]
