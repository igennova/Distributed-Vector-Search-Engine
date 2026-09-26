#!/usr/bin/env sh
# Regenerate the gRPC Python code from the .proto contract. Run from the repo root.
set -e
python -m grpc_tools.protoc -I . \
    --python_out=. --pyi_out=. --grpc_python_out=. \
    vsearch/protos/shard.proto
