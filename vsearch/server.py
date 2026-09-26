"""
A shard as a network service: one process owns one shard and serves Add and Search
over gRPC.

Run:  python -m vsearch.server --port 50051 --seed 0
"""
import argparse
import threading
from concurrent import futures

import grpc
import numpy as np

from .cluster import Shard
from .protos import shard_pb2, shard_pb2_grpc


class ShardServicer(shard_pb2_grpc.ShardServiceServicer):
    def __init__(self, **hnsw_params):
        self.shard = Shard(**hnsw_params)
        # Insert mutates the HNSW graph, so a search must not run in the middle of one.
        # One lock per shard keeps that safe; requests to a shard run one at a time.
        self.lock = threading.Lock()

    def Add(self, request, context):
        vectors = np.asarray(request.values, dtype=np.float32).reshape(-1, request.dim)
        with self.lock:
            for global_id, vector in zip(request.global_ids, vectors):
                self.shard.add(global_id, vector)
            return shard_pb2.AddResponse(size=len(self.shard))

    def Search(self, request, context):
        query = np.asarray(request.query, dtype=np.float32)
        with self.lock:
            hits = self.shard.search(query, request.k)
        return shard_pb2.SearchResponse(
            hits=[shard_pb2.Hit(global_id=global_id, distance=dist) for dist, global_id in hits])


def start_server(port=0, host="127.0.0.1", **hnsw_params):
    """Start serving one shard in the background. Returns (server, bound_port).

    port=0 lets the OS pick a free port. host defaults to loopback so the unauthenticated
    service is not exposed to the network; pass host="0.0.0.0" to accept remote clients.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    shard_pb2_grpc.add_ShardServiceServicer_to_server(ShardServicer(**hnsw_params), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, bound_port


def main():
    parser = argparse.ArgumentParser(description="Serve one shard over gRPC.")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    server, port = start_server(args.port, args.host, M=args.M,
                                ef_construction=args.ef_construction,
                                ef_search=args.ef_search, seed=args.seed)
    print(f"shard serving on {args.host}:{port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
