"""
HNSW (Hierarchical Navigable Small World) index.

Graph representation:
  self._data       : np.ndarray (capacity, dim) float32 -- unit-length vectors, one row per
                     node; only the first self._count rows are in use
  self.graph       : list; self.graph[layer] is {node_id: [neighbor_ids]}
  self.entry_point : node id where a search starts (top of the graph)
  self.top_layer   : highest layer index that exists

Vectors are normalized once when inserted, so cosine distance is just 1 - dot product, and
the distances from a query to a whole neighbor list come from one matrix-vector product.
Internal search methods expect a normalized query; the public ones normalize it themselves.
"""
import heapq
import numpy as np

_INITIAL_CAPACITY = 16
NEIGHBOR_SELECTION = ("heuristic", "closest")


def _normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector   # a zero vector ends up at distance 1 from all


class HNSW:
    def __init__(self, M=16, ef_construction=200, ef_search=50, seed=None,
                 neighbor_selection="heuristic"):
        if neighbor_selection not in NEIGHBOR_SELECTION:
            raise ValueError(f"neighbor_selection must be one of {NEIGHBOR_SELECTION}")
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.neighbor_selection = neighbor_selection
        self.mL = 1.0 / np.log(M)          # level-generation scale (exponential decay)
        self.rng = np.random.default_rng(seed)
        self._data = None                  # (capacity, dim) unit vectors, grown by doubling
        self._count = 0
        self.graph = []                    # list of {node_id: [neighbor_ids]} per layer
        self.entry_point = None
        self.top_layer = -1

    @property
    def vectors(self):
        """The stored vectors (unit length), one row per node."""
        if self._data is None:
            return np.zeros((0, 0), dtype=np.float32)
        return self._data[:self._count]

    @vectors.setter
    def vectors(self, rows):
        self.set_vectors(rows)

    def set_vectors(self, rows, normalized=False):
        """Replace the stored vectors. Pass normalized=True for rows that already have unit
        length (e.g. read back from a snapshot), so they are kept bit for bit."""
        rows = np.asarray(rows, dtype=np.float32)
        if rows.size == 0:
            self._data, self._count = None, 0
            return
        if not normalized:
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            rows = rows / norms
        self._data = np.array(rows, dtype=np.float32)
        self._count = len(rows)

    def _append(self, unit_vector):
        """Store one vector and return its node id. Doubling the capacity when full keeps the
        average cost of an append constant, like a Python list."""
        if self._data is None:
            self._data = np.empty((_INITIAL_CAPACITY, len(unit_vector)), dtype=np.float32)
        elif self._count == len(self._data):
            grown = np.empty((2 * len(self._data), self._data.shape[1]), dtype=np.float32)
            grown[:self._count] = self._data[:self._count]
            self._data = grown
        self._data[self._count] = unit_vector
        self._count += 1
        return self._count - 1

    def _distance(self, a, b):
        """Cosine distance between any two vectors: 0 = same direction, larger = farther apart."""
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
        current_dist = 1.0 - float(self._data[current] @ query)

        while True:
            neighbors = self.graph[layer][current]
            if not neighbors:
                return current
            dists = 1.0 - self._data[neighbors] @ query    # all neighbors in one product
            best = int(np.argmin(dists))

            # No neighbor is closer than where we stand -> local minimum, stop.
            if dists[best] >= current_dist:
                return current

            current, current_dist = neighbors[best], float(dists[best])

    def _search_layer_with_distances(self, query, entry, layer, ef):
        """
        Beam search within a single layer: explore from `entry`, always expanding the
        closest unexplored node, while keeping the `ef` closest nodes found so far.
        Keeping a set of candidates (rather than a single node) lets it explore around
        local minima. Returns the ef closest nodes as (distance, id) pairs (unordered).

        candidates : min-heap by distance      -> next node to expand is the closest one
        results    : max-heap stored as (-distance, id) -> the farthest kept node is on
                     top, so it is cheap to drop when the set exceeds ef
        """
        d_entry = 1.0 - float(self._data[entry] @ query)
        visited = {entry}
        candidates = [(d_entry, entry)]
        results = [(-d_entry, entry)]
        adjacency = self.graph[layer]

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # The closest remaining candidate is farther than our worst kept result,
            # so nothing unexplored can improve the set.
            if dist_c > -results[0][0]:
                break

            fresh = [n for n in adjacency[c] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = 1.0 - self._data[fresh] @ query        # one product per expanded node

            for neighbor, d in zip(fresh, dists.tolist()):
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-neg_dist, node_id) for neg_dist, node_id in results]

    def _search_layer(self, query, entry, layer, ef):
        """Like _search_layer_with_distances, but returns only the node ids."""
        return [node_id for _, node_id in self._search_layer_with_distances(query, entry, layer, ef)]

    def _random_level(self):
        """Draw a node's top layer. Exponentially decaying: most nodes get level 0."""
        return int(-np.log(self.rng.random()) * self.mL)

    def _select_neighbors(self, candidate_ids, dists, m):
        """Pick up to m of the candidates as neighbors of a base node; `dists` holds each
        candidate's distance to that node.

        "closest" keeps the m nearest. "heuristic" (HNSW paper, Algorithm 4) goes from nearest
        to farthest and keeps a candidate only if it is closer to the base node than to every
        neighbor kept so far. A rejected candidate can be reached through a kept neighbor
        anyway, so its slot goes to a link pointing somewhere new; that keeps clusters
        connected to each other instead of only to themselves. It may return fewer than m:
        filling the leftover slots with rejected candidates measured as slower to build and
        search for almost no recall.
        """
        order = np.argsort(dists, kind="stable")
        if len(order) <= m or self.neighbor_selection == "closest":
            return [candidate_ids[i] for i in order[:m]]

        ids = [candidate_ids[i] for i in order]
        vectors = self._data[ids]
        between = 1.0 - vectors @ vectors.T            # candidate-to-candidate distances
        nearest_kept = np.full(len(ids), np.inf, dtype=np.float32)
        kept = []
        for i, to_base in enumerate(dists[order].tolist()):
            if to_base < nearest_kept[i]:
                kept.append(ids[i])
                if len(kept) == m:
                    break
                np.minimum(nearest_kept, between[i], out=nearest_kept)
        return kept

    def insert(self, vector):
        """Add one vector to the index, wiring it into the graph at every layer it occupies."""
        unit = _normalize(np.asarray(vector, dtype=np.float32))
        node_id = self._append(unit)
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
            entry = self._greedy_descend(unit, entry, layer)

        # Phase B — from this node's level down to 0, find neighbors and connect.
        for layer in range(min(level, self.top_layer), -1, -1):
            scored = self._search_layer_with_distances(unit, entry, layer, self.ef_construction)
            scored.sort(key=lambda pair: pair[0])           # the search already has the distances
            neighbors = self._select_neighbors(
                [candidate for _, candidate in scored],
                np.array([dist for dist, _ in scored], dtype=np.float32), self.M)
            max_conn = self.M * 2 if layer == 0 else self.M   # layer 0 gets a higher cap

            for n in neighbors:
                self.graph[layer][node_id].append(n)
                links = self.graph[layer][n]
                links.append(node_id)
                # Keep neighbor degree bounded so search stays cheap.
                if len(links) > max_conn:
                    self.graph[layer][n] = self._select_neighbors(
                        links, 1.0 - self._data[links] @ self._data[n], max_conn)

            entry = scored[0][1]

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
        query = _normalize(np.asarray(query, dtype=np.float32))
        entry = self.entry_point
        for layer in range(self.top_layer, 0, -1):        # descend the sparse upper layers
            entry = self._greedy_descend(query, entry, layer)
        scored = self._search_layer_with_distances(query, entry, 0, self.ef_search)
        return sorted(scored)[:k]

    def search(self, query, k=10):
        """Return the ids of the approximate k nearest neighbors of query."""
        return [node_id for _, node_id in self.search_with_distances(query, k)]
