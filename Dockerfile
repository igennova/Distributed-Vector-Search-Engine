# One shard server. A cluster is several of these, each with its own volume
# (see docker-compose.yml).
FROM python:3.13-slim

# No .pyc files in the image, and logs reach `docker logs` as they are written.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, in their own layer, so changing the code does not reinstall them.
COPY pyproject.toml ./
RUN mkdir vsearch && touch vsearch/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf vsearch build *.egg-info
COPY vsearch ./vsearch
RUN pip install --no-cache-dir --no-deps .

# Run as an ordinary user; /data holds the shard's snapshot and write-ahead log.
RUN useradd --create-home --uid 10001 vsearch && mkdir /data && chown vsearch /data
USER vsearch
VOLUME /data
EXPOSE 50051

HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "from vsearch.client import wait_until_ready; wait_until_ready('127.0.0.1:50051', timeout=2)"

# 0.0.0.0 so other containers and published ports can reach it. The service has no
# authentication: docker-compose.yml publishes the ports on 127.0.0.1 only.
ENTRYPOINT ["python", "-m", "vsearch.server", "--host", "0.0.0.0", "--port", "50051", "--data-dir", "/data"]
CMD ["--seed", "0"]
