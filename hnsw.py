"""
HNSW (Hierarchical Navigable Small World) index.

Graph representation:
  self.vectors     : np.ndarray (n, dim)          -- the data
  self.graph       : list; self.graph[layer] is {node_id: [neighbor_ids]}
  self.entry_point : node id where a search starts (top of the graph)
  self.top_layer   : highest layer index that exists
"""
import heapq
import numpy as np


class HNSW:
    def __init__(self, M=16, ef_construction=200, ef_search=50):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.vectors = None
        self.graph = []           # list of {node_id: [neighbor_ids]} per layer
        self.entry_point = None
        self.top_layer = -1

    def _distance(self, a, b):
        """Cosine distance: 0 = identical direction, larger = farther apart."""
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return 1.0 - cos

    def _greedy_descend(self, query, entry, layer):
        """
        Greedy walk within a single layer: from `entry`, repeatedly hop to the neighbor
        closest to `query`, stopping when no neighbor is closer than the current node.
        Returns the id of the closest node reached (a local minimum).

        Used to descend the sparse upper layers and to seed the layer-0 search.
        """
        current = entry
        current_dist = self._distance(query, self.vectors[current])

        while True:
            best_neighbor = None
            best_dist = current_dist

            for neighbor in self.graph[layer][current]:
                d = self._distance(query, self.vectors[neighbor])
                if d < best_dist:
                    best_neighbor, best_dist = neighbor, d

            # No neighbor is closer than where we stand -> local minimum, stop.
            if best_neighbor is None:
                return current

            current, current_dist = best_neighbor, best_dist

    def _search_layer(self, query, entry, layer, ef):
        """
        Beam search within a single layer: explore from `entry`, always expanding the
        closest unexplored node, while keeping the `ef` closest nodes found so far.
        Keeping a set of candidates (rather than a single node) lets it explore around
        local minima. Returns the ef closest node ids in this layer (unordered).

        candidates : min-heap by distance      -> next node to expand is the closest one
        results    : max-heap stored as (-distance, id) -> the farthest kept node is on
                     top, so it is cheap to drop when the set exceeds ef
        """
        d_entry = self._distance(query, self.vectors[entry])
        visited = {entry}
        candidates = [(d_entry, entry)]
        results = [(-d_entry, entry)]

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            dist_farthest = -results[0][0]

            # The closest remaining candidate is farther than our worst kept result,
            # so nothing unexplored can improve the set.
            if dist_c > dist_farthest:
                break

            for neighbor in self.graph[layer][c]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance(query, self.vectors[neighbor])
                dist_farthest = -results[0][0]

                if len(results) < ef or d < dist_farthest:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [node_id for (_, node_id) in results]

    def search(self, query, k=10):
        raise NotImplementedError

    def insert(self, vector):
        raise NotImplementedError
