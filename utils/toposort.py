class Vertex:
    def __init__(self, idx):
        self.idx = idx
        self.in_deg = 0
        self.is_active = True


class Graph:
    def __init__(self):
        self.V: set[Vertex] = set()
        self.adj: dict[Vertex, set[Vertex]] = dict()
        self.dp: dict[frozenset[Vertex], int] = dict()

    def add_edge(self, u: Vertex, v: Vertex) -> None:
        self.V.add(u)
        self.V.add(v)
        self.adj.get(u, set()).add(v)
        v.in_deg += 1

    def get_sources(self) -> frozenset[Vertex]:
        sources = frozenset()
        for v in self.V:
            if v.is_active and v.in_deg == 0:
                print(f"v = {v.idx}")
                sources = sources.union({v})
        return sources

    def add_vertex(self, u: Vertex):
        self.V.add(u)
        u.is_active = True

    def delete_vertex(self, u: Vertex):
        self.V.remove(u)
        u.is_active = False

    def count(self, sources: frozenset[Vertex]) -> int:
        if len(self.V) <= 1:
            return 1
        if len(sources) == 0:
            return 0
        if sources in self.dp:
            return self.dp[sources]
        aux = sources
        res = 0
        for u in sources:
            # Remove u
            aux = aux.difference({u})
            self.delete_vertex(u)
            for v in self.adj.get(u, set()):
                if v.is_active:
                    v.in_deg -= 1
                    if v.in_deg == 0:
                        aux = aux.union({v})

            # Recursively calculate count on subgraph without u
            res += self.count(aux)

            # Recover u
            aux = aux.union({u})
            self.add_vertex(u)
            for v in self.adj.get(u, set()):
                if v.is_active:
                    if v.in_deg == 0:
                        aux = aux.difference({v})
                    v.in_deg += 1

        # Memoize for efficiency
        self.dp[sources] = res
        return res


def main():
    v1 = Vertex(1)
    v2 = Vertex(2)
    v3 = Vertex(3)
    G = Graph()
    G.add_edge(v1, v2)
    G.add_edge(v1, v3)
    srcs = G.get_sources()

    print("srcs = { ", end="")
    for k in srcs:
        print(f"{k.idx} ", end="")
    print("}")

    count = G.count(srcs)
    print(f"count = {count}")


if __name__ == "__main__":
    a = frozenset()
    main()
