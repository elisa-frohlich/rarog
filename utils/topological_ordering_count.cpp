// Following the ideas behind this site: https://math.stackexchange.com/questions/3220701/is-it-possible-to-calculate-how-many-topological-sortings-exist-for-any-given-gr#3221401

#include <bits/stdc++.h>
using namespace std;

struct Vertex {
    Vertex(int idx) : idx{idx}, in_deg{0}, is_active{true} {}

    bool operator<(const Vertex &u) const {return idx < u.idx;}

    int idx;
    int in_deg;
    bool is_active;
};

struct Graph {
    Graph() {}

    void add_edge(Vertex *u, Vertex *v) {
        if (!V.count(u)) V.insert(u);
        if (!V.count(v)) V.insert(v);

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

    int count(set<Vertex*> sources) {
        if (V.size() <= 1) return 1;
        if (sources.empty()) return dp[sources] = 0;

        if (dp.count(sources)) return dp[sources];

        set<Vertex*> aux = sources;

        int res = 0;
        for (Vertex* u : sources) {
            aux.erase(u);
            delete_vertex(u);
            for (Vertex* v : adj[u]) {
                if (v->is_active) {
                    --v->in_deg;
                    if (v->in_deg == 0) aux.emplace(v);
                }
            }

            res += count(aux);
            
            add_vertex(u);
            for (Vertex* v : adj[u]) {
                if (v->is_active) {
                    if (v->in_deg == 0) aux.erase(v);
                    ++v->in_deg;
                }
            }
            aux.emplace(u);
        }

        return dp[sources] = res;
    }

    set<Vertex*> get_sources() {
        set<Vertex*> sources;
        for (Vertex *v : V) {
            if (v->is_active && v->in_deg == 0) sources.emplace(v);
        }
        return sources;
    }

    set<Vertex*> V;
    map<Vertex*, set<Vertex*>> adj;
    map<set<Vertex*>, int> dp;
};

int main() {
    Vertex v1(0), v2(1), v3(2), v4(3), v5(4);
    Graph G;
    G.add_edge(&v1, &v2);
    // G.add_edge(&v1, &v4);
    G.add_edge(&v3, &v2);
    G.add_edge(&v3, &v4);
    G.add_edge(&v2, &v5);
    // G.add_edge(&v4, &v5);

    cout << G.count(G.get_sources()) << endl;
}