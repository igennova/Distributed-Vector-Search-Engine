#!/usr/bin/env sh
# Smoke test of the Docker setup: build and start the cluster, use it from this machine,
# have Docker restart a server, check that nothing was lost, then remove everything.
# Run from the repo root, with the vsearch command installed (pip install -e .).
set -eu
# A project of its own: its containers and volumes are separate from any cluster started
# with plain `docker compose up`, which this script never touches. It needs ports
# 50051-50054 to be free.
export COMPOSE_PROJECT_NAME=vsearch-smoke
WORK=$(mktemp -d)
CLUSTER="$WORK/cluster"
trap 'docker compose down --volumes >/dev/null 2>&1; rm -rf "$WORK"' EXIT

docker compose up --detach --build --wait

vsearch --home "$CLUSTER" cluster attach \
    127.0.0.1:50051=shard-0-replica-0:50051,127.0.0.1:50052=shard-0-replica-1:50051 \
    127.0.0.1:50053=shard-1-replica-0:50051,127.0.0.1:50054=shard-1-replica-1:50051

printf '# Storage\n\nEvery write goes to the log first.\n\n# Search\n\nQueries fan out to every shard.\n' \
    > "$WORK/notes.md"
vsearch --home "$CLUSTER" ingest "$WORK/notes.md" --embedder hash
vsearch --home "$CLUSTER" search "where does a write go first" -k 1 | grep -q "notes.md > Storage"

# Docker restarts a server; it comes back from its own volume with everything it had.
docker compose restart shard-0-replica-0
docker compose up --detach --wait
vsearch --home "$CLUSTER" ops check
vsearch --home "$CLUSTER" search "where does a write go first" -k 1 | grep -q "notes.md > Storage"

echo "docker smoke test passed"
