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
    def __init__(self, M=16, ef_construction=200, ef_search=50, seed=None):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / np.log(M)          # level-generation scale (exponential decay)
        self.rng = np.random.default_rng(seed)
        self.vectors = []                  # grows one vector per insert
        self.graph = []                    # list of {node_id: [neighbor_ids]} per layer
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

    def _random_level(self):
        """Draw a node's top layer. Exponentially decaying: most nodes get level 0."""
        return int(-np.log(self.rng.random()) * self.mL)

    def _select_neighbors(self, base_vector, candidate_ids, m):
        """Pick the m candidates closest to base_vector."""
        ordered = sorted(candidate_ids, key=lambda n: self._distance(base_vector, self.vectors[n]))
        return ordered[:m]

    def insert(self, vector):
        """Add one vector to the index, wiring it into the graph at every layer it occupies."""
        vector = np.asarray(vector, dtype=np.float32)
        node_id = len(self.vectors)
        self.vectors.append(vector)
        level = self._random_level()

        # Make sure the graph has enough layers, and register this node (no links yet).
        while len(self.graph) <= level:
            self.graph.append({})
        for layer in range(level + 1):
            self.graph[layer][node_id] = []

        # First node ever: it is the whole graph.
        if self.entry_point is None:
            self.entry_point = node_id
            self.top_layer = level
            return

        # Phase A — descend the layers above this node's level, navigation only.
        entry = self.entry_point
        for layer in range(self.top_layer, level, -1):
            entry = self._greedy_descend(vector, entry, layer)

        # Phase B — from this node's level down to 0, find neighbors and connect.
        for layer in range(min(level, self.top_layer), -1, -1):
            candidates = self._search_layer(vector, entry, layer, self.ef_construction)
            neighbors = self._select_neighbors(vector, candidates, self.M)
            max_conn = self.M * 2 if layer == 0 else self.M   # layer 0 gets a higher cap

            for n in neighbors:
                self.graph[layer][node_id].append(n)
                self.graph[layer][n].append(node_id)
                # Keep neighbor degree bounded so search stays cheap.
                if len(self.graph[layer][n]) > max_conn:
                    self.graph[layer][n] = self._select_neighbors(
                        self.vectors[n], self.graph[layer][n], max_conn)

            entry = min(candidates, key=lambda n: self._distance(vector, self.vectors[n]))

        # A taller-than-current node becomes the new entry point.
        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = node_id

    def build(self, vectors):
        """Insert many vectors. Returns self for chaining."""
        for v in vectors:
            self.insert(v)
        return self

    def search_with_distances(self, query, k=10):
        """Return the approximate k nearest neighbors as (distance, id) pairs, closest first."""
        if self.entry_point is None:                      # empty index
            return []
        query = np.asarray(query, dtype=np.float32)
        entry = self.entry_point
        for layer in range(self.top_layer, 0, -1):        # descend the sparse upper layers
            entry = self._greedy_descend(query, entry, layer)
        candidates = self._search_layer(query, entry, 0, self.ef_search)
        scored = sorted((self._distance(query, self.vectors[n]), n) for n in candidates)
        return scored[:k]

    def search(self, query, k=10):
        """Return the ids of the approximate k nearest neighbors of query."""
        return [node_id for _, node_id in self.search_with_distances(query, k)]
