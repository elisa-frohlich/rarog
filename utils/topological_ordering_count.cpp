// Following the ideas behind this site:
// https://math.stackexchange.com/questions/3220701/is-it-possible-to-calculate-how-many-topological-sortings-exist-for-any-given-gr#3221401

#include <bits/stdc++.h>
using namespace std;

struct Vertex {
    Vertex(int idx) : idx{idx}, in_deg{0}, is_active{true} {
    }

    bool operator<(const Vertex &u) const {
        return idx < u.idx;
    }

    int idx;
    int in_deg;
    bool is_active;
};

struct Graph {
    Graph() {
    }

    void add_edge(Vertex *u, Vertex *v) {
        if (!V.count(u))
            V.insert(u);
        if (!V.count(v))
            V.insert(v);

        if (!adj[u].count(v)) {
            adj[u].emplace(v);
            ++v->in_deg;
        }
    }

    void add_vertex(Vertex *u) {
        V.emplace(u);
        u->is_active = true;
    }

    void delete_vertex(Vertex *u) {
        V.erase(u);
        u->is_active = false;
    }

    int count(set<Vertex *> sources) {
        // If G has 0 vertices, it has exactly 1 topological sorting.
        if (V.size() <= 1)
            return 1;

        // Otherwise...  Find the source vertices of G. (These are just the
        // vertices with indegree 0.)

        // If there are none, there are no topological sortings of G.
        if (sources.empty())
            return dp[sources] = 0;

        if (dp.count(sources))
            return dp[sources];

        // For each source vertex s of G, let ts be the number of topological
        // sortings of the simpler graph Gs obtained by deleting s from G.
        set<Vertex *> aux = sources;

        // The  number you want is just the sum of the ts values.
        int res = 0;

        for (Vertex *s : sources) {

            // Remove s from the graph
            aux.erase(s);
            delete_vertex(s);
            // Recalculate what the sources are after removing s
            for (Vertex *v : adj[s]) {
                if (v->is_active) {
                    --v->in_deg;
                    if (v->in_deg == 0)
                        aux.emplace(v);
                }
            }

            // Recursively calculate topo sort of smaller graph
            res += count(aux);

            // Restore s in the graph
            add_vertex(s);
            // Restore the in_deg of vertices and remove sources pointed by s
            for (Vertex *v : adj[s]) {
                if (v->is_active) {
                    if (v->in_deg == 0)
                        aux.erase(v);
                    ++v->in_deg;
                }
            }
            aux.emplace(s);
        }

        return dp[sources] = res;
    }

    set<Vertex *> get_sources() {
        set<Vertex *> sources;
        for (Vertex *v : V) {
            if (v->is_active && v->in_deg == 0)
                sources.emplace(v);
        }
        return sources;
    }

    set<Vertex *> V;
    map<Vertex *, set<Vertex *>> adj;
    map<set<Vertex *>, int> dp;
};

/*
Inspired by:

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @tf2onnx(%1: tensor<?x32x32x3xf32>) -> tensor<?x10xf32> {
    %0 = arith.constant 0 : index
    %5 = arith.constant 0.000000e+00 : f32
    %2 = tensor.dim %1, %0 : tensor<?x32x32x3xf32>
    %3 = tensor.empty(%2) : tensor<?x3x32x32xf32>
    %4 = tensor.empty(%2) : tensor<?x10xf32>
    %6 = linalg.fill ins(%5 : f32) outs(%4 : tensor<?x10xf32>) ->
        tensor<?x10xf32>
    return %6 : tensor<?x10xf32>
  }
}
*/
int main() {
    // Vertex v0(0), v1(1), v2(2), v3(3), v4(4), v5(5), v6(6);
    // Graph G;
    // G.add_edge(&v0, &v2);
    // G.add_edge(&v1, &v2);
    // G.add_edge(&v2, &v3);
    // G.add_edge(&v2, &v4);
    // G.add_edge(&v4, &v6);
    // G.add_edge(&v5, &v6);

    Vertex v0(0), v1(1), v2(2), v3(3);
    Graph G;
    G.add_edge(&v0, &v2);
    G.add_edge(&v1, &v2);
    G.add_edge(&v3, &v0);

    auto srcs = G.get_sources();
    cout << "[ ";
    for (auto src : srcs) {
        cout << src->idx << " ";
    }
    cout << "]" << endl;

    cout << G.count(G.get_sources()) << endl;

    cout << "And what is in dp:" << endl;

    for (auto &[srcs, count] : G.dp) {

        cout << "topos(srcs={ ";
        for (auto src : srcs)
            cout << src->idx << " ";
        cout << "}) = " << count << endl;
    }
}