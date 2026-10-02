// === UTILS and LIS ===

// Fast min/max assignment
// Returns 1 if 'a' was updated, 0 otherwise
template<class T> bool chmin(T& a, const T& b) {
    return b < a ? a = b, 1 : 0;
}

template<class T> bool chmax(T& a, const T& b) {
    return a < b ? a = b, 1 : 0;
}

// Longest Increasing Subsequence
// - Computes STRICTLY increasing subsequence.
// - For NON-DECREASING (allowing duplicates), change lower_bound -> upper_bound.
// - Returns length only; to reconstruct sequence, maintain parent pointers.
int lis(const vi& a) {
    vi dp;
    trav(v, a) {
        int pos = lower_bound(all(dp), v) - dp.begin();
        if (pos == sz(dp)) dp.pb(v);
        else dp[pos] = v;
    }
    return sz(dp);
}

// Random number generator seeded with steady_clock (unhackable in contests)
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());

// Fisher-Yates partial shuffle
// - Mutates 'popl' in-place.
// - 'res' must be pre-allocated to size k.
auto fisher_yates = [](vi& popl, vi& res, int k) -> void {
    rep(i, 0, k) {
        uniform_int_distribution<> dist(i, sz(popl) - 1);
        int j = dist(rng);
        swap(popl[i], popl[j]);
    }
    rep(i, 0, k) res[i] = popl[i];
};

// Full shuffle
// - Use std::shuffle with rng; avoid std::random_shuffle (deprecated, predictable).
// shuffle(all(arr), rng);

// === GRAPH TRAVERSAL ===

// Flood Fill BFS (0-indexed grid [0, n-1] x [0, m-1])
// - Must mark vis[xx][yy] = true immediately upon pushing to queue to avoid duplicate queue states.
// - Grid bounds check uses IN(xx, 0, n - 1) && IN(yy, 0, m - 1).
int dx[4] = {1, -1, 0, 0};
int dy[4] = {0, 0, 1, -1};
queue<pi> bfs;
while (!bfs.empty()) {
    pi c = bfs.front(); bfs.pop();
    // do some function on c
    rep(k, 0, 4) {
        int xx = c.fr + dx[k];
        int yy = c.se + dy[k];
        if (IN(xx, 0, n - 1) && IN(yy, 0, m - 1) && !vis[xx][yy]) {
            vis[xx][yy] = true;
            bfs.push({xx, yy});
        }
    }
}

// Flood Fill DFS: same idea, adj[u] becomes neighboring grid cells

// Functional Graph Cycle Detection (Floyd's Tortoise and Hare)
// - Assumes functional graph (out-degree = 1 for every node via succ(u)).
// - 'start' must be a valid node in the component.
// - Result: mu = path length to enter cycle, lambda = cycle length.
void floyd(int start = 0) {
    // 1. Find meeting point inside cycle
    int a = succ(start);
    int b = succ(succ(start));
    while (a != b) {
        a = succ(a);
        b = succ(succ(b));
    }
    // 2. Find start of cycle (mu)
    a = start;
    int mu = 0;
    while (a != b) {
        a = succ(a);
        b = succ(b);
        ++mu;
    }
    // 3. Find cycle length (lambda)
    int lambda = 1;
    b = succ(a);
    while (a != b) {
        b = succ(b);
        ++lambda;
    }
    // start of cycle = mu, cycle length = lambda
}

// Topo Sort - DFS
// - 1-indexed graph.
// - Assumes graph is already a DAG (does NOT detect cycles). If cycles might exist, use Kahn's BFS.
vi order;
void dfs(int u) {
    vis[u] = true;
    trav(v, adj[u]) {
        if (!vis[v]) dfs(v);
    }
    order.pb(u);
}

void topo_sort() {
    order.clear();
    vis.assign(n + 1, false);
    rep(i, 1, n + 1) {
        if (!vis[i]) dfs(i);
    }
    reverse(all(order)); // order contains toposort
}

// Topo Sort - BFS (Kahn's algorithm)
// - 1-indexed graph.
// - Detects cycles: if sz(order) != n, graph has a directed cycle (no valid topo sort).
// - Use priority_queue<int, vi, greater<int>> instead of queue to obtain the lexicographically smallest topo sort.
queue<int> q;
rep(i, 1, n + 1) {
    if (indegree[i] == 0) q.push(i);
}
vi order;
while (!q.empty()) {
    int c = q.front(); q.pop();
    order.pb(c);
    trav(v, adj[c]) {
        if (--indegree[v] == 0) q.push(v);
    }
}
if (sz(order) != n) {
    // Graph has a cycle (no valid topo sort)
} else {
    // order contains valid topo sort
}

// === NUMBER THEORY ===

// Prime Factorization Sieve: spf = smallest prime factor
// - spf[x] stores the smallest prime factor dividing x.
// - spf[0] = spf[1] = 0 (neither is prime).
// - Factorize x: while (x > 1) { int p = spf[x]; while (x % p == 0) x /= p; }
int spf[U];
void sieve_spf() {
    rep(i, 2, U) if (!spf[i]) {
        spf[i] = i;
        for (ll j = (ll)i * i; j < U; j += i) {
            if (!spf[j]) spf[j] = i;
        }
    }
}

// Binary Exponentiation: (a^b) % M
// - Handles a >= M via initial a %= M. Works for b = 0 (returns 1 % M).
ll exp(ll a, ll b) {
    ll res = 1;
    a %= M;
    while (b) {
        if (b & 1) res = (res * a) % M;
        a = (a * a) % M;
        b >>= 1;
    }
    return res;
}

// Modular Inverse
// - Modulo M MUST be prime!
// - 'i' must not be a multiple of M (i % M != 0).
// - If M is composite or gcd(i, M) == 1, use ext_gcd instead.
ll inv(ll i) {
    return i <= 1 ? i : M - (ll)(M / i) * inv(M % i) % M;
}

// Euler Totient (Single Value)
// - Defined for n >= 1. phi(1) = 1.
int phi(int n) {
    int ans = n;
    for (int p = 2; (ll)p * p <= n; ++p) {
        if (n % p == 0) {
            while (n % p == 0) n /= p;
            ans -= ans / p;
        }
    }
    if (n > 1) ans -= ans / n;
    return ans;
}

// Extended Euclidean: a*x + b*y = gcd(a, b)
// - 'x' and 'y' are passed by reference and can be negative.
// - To find modular inverse of 'a' modulo 'm' (when gcd(a, m) = 1):
//     ext_gcd(a, m, x, y);
//     ll a_inv = (x % m + m) % m;
ll ext_gcd(ll a, ll b, ll& x, ll& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    ll xx, yy;
    ll g = ext_gcd(b, a % b, xx, yy);
    x = yy;
    y = xx - yy * (a / b);
    return g;
}

// Combinations (nCr) modulo M
// - M MUST be prime and M > U.
// - Must call precompute_comb() once before querying C(n, r).
// - Returns 0 for r < 0 or r > n.
ll F[U], iF[U];
void precompute_comb() {
    F[0] = F[1] = 1;
    rep(i, 2, U) F[i] = (F[i - 1] * i) % M;

    iF[U - 1] = inv(F[U - 1]);
    rrep(i, U - 2, -1) iF[i] = (iF[i + 1] * (i + 1)) % M;
}

ll C(int n, int r) {
    if (r < 0 || r > n) return 0;
    return F[n] * iF[r] % M * iF[n - r] % M;
}

// Lucas Theorem: nCr mod p
// - 'p' MUST be a small prime (typically p <= 1e6).
// - Factorials F and iF must be precomputed modulo p up to p - 1.
// - Supports n, r up to 1e18.
ll lucas(ll n, ll r, ll p) {
    ll res = 1;
    while (n) {
        ll N = n % p;
        ll R = r % p;

        if (R > N) return 0;
        ll cur = 1;
        if (R != 0) cur = (F[N] * iF[R] % p) * iF[N - R] % p;

        res = (res * cur) % p;
        n /= p;
        r /= p;
    }
    return res;
}

// Euler Totient Sieve
// - Precomputes phi values for all integers 0..n. phi(0) = 0, phi(1) = 1.
vi get_phi(int n) {
    vi phi_arr(n + 1);
    iota(all(phi_arr), 0);
    rep(i, 2, n + 1) {
        if (phi_arr[i] == i) {
            for (int j = i; j <= n; j += i) {
                phi_arr[j] -= phi_arr[j] / i;
            }
        }
    }
    return phi_arr;
}

// === SHORTEST PATHS ===

// 0-1 BFS
// - Edge weights MUST ONLY be 0 or 1.
// - Push 0-weight edges to FRONT, 1-weight edges to BACK.
// - 1-indexed graph. Do NOT mark visited; relax distance directly.
vll d(n + 1, LLONG_MAX);
d[root] = 0;
deque<int> q;
q.push_front(root);
while (!q.empty()) {
    int c = q.front(); q.pop_front();
    trav(edge, adj[c]) {
        auto [v, w] = edge;
        if (d[v] > d[c] + w) {
            d[v] = d[c] + w;
            if (w) q.pb(v);
            else q.push_front(v);
        }
    }
}

// Dijkstra (adj contains {v, weight})
// - NO negative edge weights allowed!
// - Skip outdated queue entries (cdist != dist[node]) to maintain performance.
// - Uses ll for distances to avoid 32-bit integer overflow.
vll dist(n + 1, LLONG_MAX);
using T = pair<ll, int>;
priority_queue<T, vt<T>, greater<T>> pq;
dist[1] = 0;
pq.push({0, 1});
while (!pq.empty()) {
    auto [cdist, node] = pq.top(); pq.pop();
    if (cdist != dist[node]) continue; // node has been updated before this, skip old pair
    trav(edge, adj[node]) {
        auto [v, w] = edge;
        if (dist[v] > cdist + w) {
            dist[v] = cdist + w;
            pq.push({dist[v], v});
        }
    }
}

// Floyd-Warshall
// - The 'k' loop MUST be the outermost loop.
// - Handles negative edge weights, but NO negative cycles.
// - If dist[i][i] < 0 for any node i, a negative cycle exists containing i.
// - Condition (dist[i][k] < INF && dist[k][j] < INF) prevents signed overflow with large INF.
rep(i, 1, n + 1) {
    rep(j, 1, n + 1) dist[i][j] = INF;
    dist[i][i] = 0;
}
rep(i, 0, m) {
    int a, b; ll c; cin >> a >> b >> c;
    chmin(dist[a][b], c);
    // dist[b][a] = dist[a][b]; // if undirected
}

rep(k, 1, n + 1) {
    rep(i, 1, n + 1) {
        rep(j, 1, n + 1) {
            if (dist[i][k] < INF && dist[k][j] < INF) {
                chmin(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

// === DSU & MST ===

// Disjoint Set Union (Union-Find)
// - Supports both 0-indexed and 1-indexed elements (size N).
// - unite(x, y) returns 1 if merged, 0 if already in the same set.
// - size(x) returns the size of the component containing x.
struct DSU {
    vi e;
    DSU(int N) : e(N, -1) {}
    int get(int x) { return e[x] < 0 ? x : e[x] = get(e[x]); }
    bool same(int a, int b) { return get(a) == get(b); }
    int size(int x) { return -e[get(x)]; }
    bool unite(int x, int y) {
        x = get(x); y = get(y);
        if (x == y) return 0;
        if (e[x] > e[y]) swap(x, y);
        e[x] += e[y];
        e[y] = x;
        return 1;
    }
};

// Kruskal's MST
// - Edges MUST be sorted before running (or uncomment sort(all(edges))).
// - Graph must be undirected.
// - If sz(mst) < n - 1, the graph is disconnected (forms an MST forest).
struct Edge {
    int u, v;
    ll w;
    bool operator<(const Edge& o) const { return w < o.w; }
};

vt<Edge> edges;
// sort(all(edges));
DSU ds(n + 1);
ll cost = 0;
vt<Edge> mst;

trav(e, edges) {
    if (ds.unite(e.u, e.v)) {
        cost += e.w;
        mst.pb(e);
        if (sz(mst) == n - 1) break;
    }
}

// Prim's MST (for Dense Graphs)
// - For dense graphs only (MAXV ~ 2500); avoid large N due to V^2 matrix memory.
// - Undirected graphs require adj[u][v] = adj[v][u]. Non-existent edges should have weight INF.
// - Returns cost = -1 if the graph is disconnected.
const int MAXV = 2500;
int adj[MAXV][MAXV];
bool vis[MAXV];
vt<pi> min_e(MAXV, {M, -1});
ll cost = 0;
min_e[0].fr = 0; // 0 chosen to be root of mst
rep(i, 0, n) {
    int v = -1;
    rep(j, 0, n) {
        if (!vis[j] && (v == -1 || min_e[j].fr < min_e[v].fr)) v = j;
    }
    if (v == -1 || min_e[v].fr == M) {
        cost = -1; break; // no mst possible
    }
    vis[v] = true;
    cost += min_e[v].fr;
    if (min_e[v].se != -1) {
        // valid edge of mst formed
    }
    rep(to, 0, n) {
        if (adj[v][to] < min_e[to].fr) {
            min_e[to] = {adj[v][to], v};
        }
    }
}
// first iteration adds root node for cost 0, rest all start adding edges of tree
// output -> cost
// same idea used for dijkstra with dense graphs

// === RANGE QUERIES ===

// Fenwick Tree (Binary Indexed Tree)
// - Strictly 1-indexed. add(0, ...) enters an infinite loop!
// - Point update, range sum query over [l, r].
struct FenwickTree {
    vt<ll> bit;
    int n;

    // O(n) construction
    void init(const vt<ll>& a) {
        n = sz(a);
        bit.assign(n, 0);
        rep(i, 1, n) {
            bit[i] += a[i];
            int r = i + (i & (-i));
            if (r < n) bit[r] += bit[i];
        }
    }

    ll sum(int r) {
        ll ret = 0;
        for (; r > 0; r -= r & (-r)) ret += bit[r];
        return ret;
    }

    ll sum(int l, int r) {
        return sum(r) - sum(l - 1);
    }

    void add(int x, ll delta) {
        for (; x < n; x += x & (-x)) bit[x] += delta;
    }
    // check cses/dynrangesum for updates
};

// Segment Tree (Iterative, Point Update, Range Query)
// - 0-indexed array with half-open query intervals [l, r).
// - 'combine' function must be associative with an identity element (e.g. 0 for sum, INF for min).
// - 'walk(x)' requires n to be padded to a power of 2.
const int MAXN = 2e5 + 5;
node seg[2 * MAXN];
int n;

void build(const vt<node>& a) {
    rep(i, 0, n) seg[n + i] = a[i];
    rrep(i, n - 1, 0) seg[i] = combine(seg[2 * i], seg[2 * i + 1]);
}

void upd(int p, node v) {
    p += n;
    seg[p] = v;
    while (p > 1) {
        p >>= 1;
        seg[p] = combine(seg[2 * p], seg[2 * p + 1]);
    }
}

// [l, r) queries
node get(int l, int r) {
    node resl, resr; // identity
    l += n; r += n;
    for (; l < r; l >>= 1, r >>= 1) {
        if (l & 1) resl = combine(resl, seg[l++]);
        if (r & 1) resr = combine(seg[--r], resr);
    }
    return combine(resl, resr);
}

// from cses/hotelqueries: first index with seg[p] >= x
// needs padding to power of 2
int walk(int x) {
    int p = 1, lo = 0, hi = n - 1;
    while (hi > lo) {
        int m = (lo + hi) / 2;
        if (seg[2 * p] >= x) {
            p = 2 * p; hi = m;
        } else {
            p = 2 * p + 1; lo = m + 1;
        }
    }
    return seg[p] >= x ? lo : -1;
}

// Order Statistic Tree and Hash table
// - find_by_order(k) is 0-indexed (k = 0 returns an iterator to the minimum element).
// - order_of_key(x) returns the count of elements strictly smaller than x.
// - chash prevents custom hash collisions / anti-hash tests in unordered maps.
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
// template <class T> using OSTree = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>; (defined in template.cpp)
OSTree<int> ost;
ost.find_by_order(1); // -> iterator for ith element 
ost.order_of_key(5);  // -> number of elements strictly lesser than 5

// custom hash for faster hash map
struct chash {
    const uint64_t C = uint64_t(2e18 * 3.14) + 71;
    const uint32_t RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const {
        return __builtin_bswap64((x ^ RANDOM) * C);
    }
};
gp_hash_table<int, int, chash> ht; // -> faster hash map

// Sparse Table: for static idempotent range queries (min, max, gcd)
// stores value of function over [i, i + 2^j) for j <= LOG_U; check cses/staticmin

// Lazy Segment Tree (Recursive, Range Update, Range Query)
// - 1-indexed tree over range [1, n]. Both updates and queries use inclusive intervals [l, r].
// - Currently configured for Range Add + Range Min.
// - To change to Range Add + Range Sum: prop() must be: seg[u] += 1LL * len * fa; and neutral element in get() must be 0.
int seg[4 * MAXN], lazy[4 * MAXN];

void build(const vi& a, int u = 1, int tl = 1, int tr = n) {
    if (tl == tr) {
        seg[u] = a[tl];
        return;
    }
    int tm = (tl + tr) / 2;
    build(a, 2 * u, tl, tm);
    build(a, 2 * u + 1, tm + 1, tr);
    seg[u] = min(seg[2 * u], seg[2 * u + 1]);
}

inline void prop(int u, int len, int fa) {
    // if min gets increased by fa, then segment min also gets increased by fa
    // in other cases like sum, handle the propagation correctly with len (e.g. seg[u] += 1LL * len * fa)
    seg[u] += fa;
    lazy[u] += fa;
}

inline void push(int u, int tl, int tr) {
    if (lazy[u] == 0) return; // range add of 0 does nothing
    int tm = (tl + tr) / 2;
    prop(2 * u, tm - tl + 1, lazy[u]);
    prop(2 * u + 1, tr - tm, lazy[u]);
    lazy[u] = 0;
}

void upd(int l, int r, int v, int u = 1, int tl = 1, int tr = n) {
    if (l > tr || r < tl) return;
    if (l <= tl && tr <= r) {
        prop(u, tr - tl + 1, v);
    } else {
        push(u, tl, tr);
        int tm = (tl + tr) / 2;
        upd(l, r, v, 2 * u, tl, tm);
        upd(l, r, v, 2 * u + 1, tm + 1, tr);
        seg[u] = min(seg[2 * u], seg[2 * u + 1]);
    }
}

int get(int l, int r, int u = 1, int tl = 1, int tr = n) {
    if (l > tr || r < tl) return 1e9; // neutral element
    if (l <= tl && tr <= r) return seg[u];
    
    push(u, tl, tr);
    int tm = (tl + tr) / 2;
    int left = get(l, r, 2 * u, tl, tm);
    int right = get(l, r, 2 * u + 1, tm + 1, tr);
    return min(left, right);
}

// walk over segment tree to find first index with value < 0
int walk(int u, int tl, int tr) {
    if (tl == tr) {
        if (seg[u] >= 0) return tr + 1;
        return tl;
    }
    push(u, tl, tr);
    int tm = (tl + tr) / 2;
    if (seg[2 * u] < 0) return walk(2 * u, tl, tm);
    return walk(2 * u + 1, tm + 1, tr);
}

// === CONNECTED COMPS ===

// Finding Bridges (undirected graph)
// - Undirected graph, 1-indexed (nodes 1..n).
// - Multi-edge warning: if the graph has parallel edges between the same two nodes,
//   checking 'v == p' treats the second edge as a back-edge and falsely misses bridges.
//   For multigraphs, pass the incoming edge ID instead of parent node 'p'.
struct Bridges {
    int n, t;
    vi tin, low;
    vt<char> vis;
    vt<pi> bridges;

    Bridges(const vvi& g) : n(sz(g) - 1), t(0), tin(n + 1, -1), low(n + 1, -1), vis(n + 1, 0) {
        auto dfs = [&](auto&& self, int u, int p) -> void {
            vis[u] = 1;
            tin[u] = low[u] = t++;
            trav(v, g[u]) {
                if (v == p) continue;
                if (vis[v]) {
                    // back-edge
                    chmin(low[u], tin[v]);
                } else {
                    self(self, v, u);
                    chmin(low[u], low[v]);
                    // low[v] > tin[u] means (u, v) is a bridge
                    if (low[v] > tin[u]) {
                        bridges.pb({u, v});
                    }
                }
            }
        };

        rep(i, 1, n + 1) {
            if (!vis[i]) dfs(dfs, i, 0);
        }
    }
};

// Finding Articulation Points (undirected graph)
// - Undirected graph, 1-indexed (nodes 1..n).
// - Root of DFS tree is an articulation point iff it has >= 2 children in the DFS tree.
// - Automatically collects deduplicated articulation points in 'art_points'.
struct ArtPoints {
    int n, t;
    vi tin, low;
    vt<char> vis, is_cut;
    vi art_points;

    ArtPoints(const vvi& g) : n(sz(g) - 1), t(0), tin(n + 1, -1), low(n + 1, -1), vis(n + 1, 0), is_cut(n + 1, 0) {
        auto dfs = [&](auto&& self, int u, int p) -> void {
            vis[u] = 1;
            tin[u] = low[u] = t++;
            int children = 0;
            trav(v, g[u]) {
                if (v == p) continue;
                if (vis[v]) {
                    chmin(low[u], tin[v]);
                } else {
                    ++children;
                    self(self, v, u);
                    chmin(low[u], low[v]);
                    if (low[v] >= tin[u] && p != 0) {
                        is_cut[u] = 1;
                    }
                }
            }
            if (p == 0 && children > 1) {
                is_cut[u] = 1;
            }
        };

        rep(i, 1, n + 1) {
            if (!vis[i]) dfs(dfs, i, 0);
        }
        rep(i, 1, n + 1) {
            if (is_cut[i]) art_points.pb(i);
        }
    }
};

// 2-Edge-Connected Components (2CC / Bridge-Block Tree)
// - Undirected graph, 1-indexed.
// - Compresses components that remain connected after removing any single edge into single nodes.
// - Equivalent to removing all bridges from the graph and condensing connected components.
// - comp_id[u] gives component ID in 1..num_comps. build_tree() builds the compressed tree.
struct TwoCC {
    struct DSU {
        vi e, h;
        const vi& depth;

        DSU(int N, const vi& d) : e(N, -1), h(N), depth(d) {
            iota(all(h), 0); // every node is the highest in its own comp
        }
        
        int get(int x) { return e[x] < 0 ? x : e[x] = get(e[x]); } 
        bool same(int a, int b) { return get(a) == get(b); }
        int size(int x) { return -e[get(x)]; }
        int highest(int x) { return h[get(x)]; }
        
        bool unite(int x, int y) {
            x = get(x); y = get(y); 
            if (x == y) return 0;
            
            // store highest node amongst both sets
            int minh = (depth[h[x]] < depth[h[y]]) ? h[x] : h[y];

            if (e[x] > e[y]) swap(x, y);
            e[x] += e[y]; 
            e[y] = x; 

            h[x] = minh; // assign highest back to merged set
            return 1;
        }
    };

    int n, num_comps;
    vi comp_id, par;
    vvi tree;

    TwoCC(const vvi& g) : n(sz(g) - 1), num_comps(0), comp_id(n + 1, -1), par(n + 1, 0) {
        vi depth(n + 1, 0);
        vt<char> vis(n + 1, 0);
        vt<pi> back_edges;
        DSU dsu(n + 1, depth); 

        auto dfs = [&](auto&& self, int u, int p, int d) -> void {
            vis[u] = 1;
            depth[u] = d;
            par[u] = p;
            
            bool skipped_parent = false;
            trav(v, g[u]) {
                if (v == p && !skipped_parent) {
                    skipped_parent = true; 
                    continue;
                }
                if (!vis[v]) {
                    self(self, v, u, d + 1);
                } else if (depth[v] < depth[u]) {
                    back_edges.pb({u, v}); 
                }
            }
        };

        rep(i, 1, n + 1) if (!vis[i]) dfs(dfs, i, 0, 1);

        trav(edge, back_edges) {
            int u = dsu.highest(edge.fr);
            int v = dsu.highest(edge.se);
            while (u != v) {
                if (depth[u] < depth[v]) swap(u, v);
                int p_node = dsu.highest(par[u]);
                dsu.unite(u, p_node);
                u = dsu.highest(u);
            }
        }

        // map sparse dsu roots to comp ids
        rep(i, 1, n + 1) {
            int r = dsu.get(i);
            if (comp_id[r] == -1) comp_id[r] = ++num_comps;
            comp_id[i] = comp_id[r];
        }
    }

    void build_tree() {
        tree.assign(num_comps + 1, {});
        rep(i, 1, n + 1) {
            if (par[i] == 0) continue; 
            int u = comp_id[i];
            int v = comp_id[par[i]];
            
            if (u != v) {
                tree[u].pb(v); 
                tree[v].pb(u);
            }
        }
    }
};

// Strongly Connected Components (Kosaraju)
// - Directed graph, 1-indexed.
// - comp_id[u] gives the SCC index (1..sz(comps)-1).
// - comps[i] contains all vertices in SCC i.
// - The condensed graph 'dag' is produced in REVERSE topological order (component with in-degree 0 in DAG has the highest index).
struct SCC {
    int n;
    vi comp_id; // comp_id[u] = ID of the SCC containing u
    vvi comps;  // comps[i] = list of nodes in SCC i
    vvi dag;    // Condensed graph of SCCs

    SCC(const vvi& g) : n(sz(g) - 1), comp_id(n + 1, 0) {
        vvi gt(n + 1);
        rep(u, 1, n + 1) {
            trav(v, g[u]) gt[v].pb(u);
        }

        vt<char> vis(n + 1, 0);
        vi order; order.reserve(n);

        // order contains nodes in increasing order of exit time
        auto dfs1 = [&](auto&& self, int u) -> void {
            vis[u] = 1;
            trav(v, g[u]) if (!vis[v]) self(self, v);
            order.pb(u);
        };

        rep(i, 1, n + 1) if (!vis[i]) dfs1(dfs1, i);

        vis.assign(n + 1, 0);
        comps.pb({}); // 0 slot unused

        auto dfs2 = [&](auto&& self, int u) -> void {
            vis[u] = 1;
            comp_id[u] = sz(comps) - 1;
            comps.back().pb(u);
            trav(v, gt[u]) if (!vis[v]) self(self, v);
        };

        trav(v, views::reverse(order)) {
            if (!vis[v]) {
                comps.pb({});
                dfs2(dfs2, v);
            }
        }

        dag.assign(sz(comps), {});
        rep(u, 1, n + 1) {
            int ru = comp_id[u];
            trav(v, g[u]) {
                int rv = comp_id[v];
                if (ru != rv) dag[ru].pb(rv);
            }
        }

        rep(i, 1, sz(comps)) {
            sort(all(dag[i]));
            auto [first, last] = ranges::unique(dag[i]);
            dag[i].erase(first, last);
        }
    }
};

// check 2SAT in cses/giantpizza

// Eulerian Cycles (Hierholzer's Algorithm)
// - Graph must be connected (excluding isolated vertices).
// - Directed graph: in-degree == out-degree for all vertices.
// - Undirected graph: all vertices must have even degree.
// - Output vectors 'nodes' and 'edges' are reversed at the end to give the correct traversal order.
vi nodes, edges;
vt<char> vis; // sized to number of edges + 1
auto dfs = [&](auto&& self, int u) -> void {
    while (!adj[u].empty()) {
        auto [to, id] = adj[u].back();
        adj[u].pop_back();
        if (vis[id]) continue;
        vis[id] = 1;
        
        self(self, to);
        edges.pb(id);
    }
    nodes.pb(u);
};

rep(i, 1, n + 1) {
    nodes.clear(); edges.clear();
    dfs(dfs, i);
    reverse(all(nodes)); reverse(all(edges));
    // nodes stores order of nodes visited in eulerian cycle
    // edges stores order of edges visited in eulerian cycle
}

// === TREES ===

// Euler Tour
// Subtree Queries - check cses/stquery
// LCA - check cses/companyqueries2
// Path Queries - check cses/pathquery

// Centroid Decomposition
// - Undirected tree, 1-indexed.
// - rem[u] = true prevents revisiting centroids already processed.
// - Depth of centroid tree is guaranteed <= log2(n).
// - When answering path queries, process/aggregate paths through centroid 'c' before setting rem[c] = true.
vt<pi> anc[U]; // anc[u] stores {centroid_ancestor, distance}
int subtree_sz[U];
bool rem[U];

int get_sz(int u, int p) {
    subtree_sz[u] = 1;
    trav(v, adj[u]) {
        if (v != p && !rem[v]) subtree_sz[u] += get_sz(v, u);
    }
    return subtree_sz[u];
}

int get_centroid(int u, int p, int total_n) {
    trav(v, adj[u]) {
        if (v == p || rem[v]) continue;
        if (subtree_sz[v] > total_n / 2) return get_centroid(v, u, total_n);
    }
    return u;
}

void find_dist(int u, int p, int c, int d) {
    trav(v, adj[u]) {
        if (v == p || rem[v]) continue;
        find_dist(v, u, c, d + 1);
    }
    anc[u].pb({c, d});
}

void build_ct(int u, int p = 0) {
    int cur_n = get_sz(u, p);
    int c = get_centroid(u, p, cur_n);
    find_dist(c, p, c, 0);
    rem[c] = true;
    trav(v, adj[c]) {
        if (rem[v]) continue;
        build_ct(v, c);
    }
}

// Binary Lifting & Lowest Common Ancestor (LCA)
// - 1-indexed tree. 'root' is typically 1.
// - jmp(x, k) returns -1 if k exceeds depth of node x.
// - dist(a, b) assumes unweighted edges (hop distance). For weighted trees, store prefix distances from root in d[u].
struct LCA {
    int n, L;
    vvi up;
    vi d;

    LCA(int n, int root, const vvi& adj) : n(n), L(ceil(log2(n + 1))), up(n + 1, vi(L + 1, 0)), d(n + 1, 0) {
        auto dfs = [&](auto&& self, int u, int p, int depth) -> void {
            d[u] = depth;
            up[u][0] = p;
            rep(i, 1, L + 1) {
                up[u][i] = up[up[u][i - 1]][i - 1];
            }
            trav(v, adj[u]) {
                if (v != p) self(self, v, u, depth + 1);
            }
        };
        dfs(dfs, root, root, 0);
    }

    int jmp(int x, int k) const {
        if (k > d[x]) return -1;
        rrep(i, L, -1) {
            if (k & (1 << i)) {
                x = up[x][i];
            }
        }
        return x;
    }

    int lca(int a, int b) const {
        if (d[a] > d[b]) swap(a, b);
        b = jmp(b, d[b] - d[a]);
        if (a == b) return a;
        rrep(i, L, -1) {
            if (up[a][i] != up[b][i]) {
                a = up[a][i];
                b = up[b][i];
            }
        }
        return up[a][0];
    }

    int dist(int a, int b) const {
        return d[a] + d[b] - 2 * d[lca(a, b)];
    }
};

// === STRINGS ===

// Polynomial Rolling Hash for Strings
// - 0-indexed string. Query get_hash(l, r) is inclusive [l, r].
// - Uses Mersenne prime (2^61 - 1) and random base B with __int128 to prevent collision hacks in contests.
class hstring {
private:
    static const ll M = (1LL << 61) - 1;
    static const ll B;
    static vll pow;
    // p_hash[i] is the hash of the first i characters of the given string
    vll p_hash;
    __int128 mul(ll a, ll b) const { return (__int128)a * b; }
    ll mod_mul(ll a, ll b) const { return mul(a, b) % M; }

public:
    hstring(const string& s) : p_hash(sz(s) + 1) {
        while (sz(pow) <= sz(s)) {
            pow.pb(mod_mul(pow.back(), B));
        }
        p_hash[0] = 0;
        rep(i, 0, sz(s)) {
            p_hash[i + 1] = (mul(p_hash[i], B) + s[i]) % M;
        }
    }

    ll get_hash(int l, int r) const {
        int w = r - l + 1;
        ll val = p_hash[r + 1] - mod_mul(p_hash[l], pow[w]);
        return (val + M) % M;
    }
};

vll hstring::pow = {1};
const ll hstring::B = uniform_int_distribution<ll>(1, (1LL << 61) - 2)(rng);

// Prefix Trie
// - Assumes lowercase English letters ('a'..'z'). Adjust ALPH and 'word[i] - a' for digits or uppercase.
// - Memory usage is proportional to (number of nodes * ALPH).
// - Node 0 is the root and represents the empty string.
const int ALPH = 26; // size of alphabet
struct node {
    int down[ALPH] = {};
    int cnt = 0;       // reference count / number of words passing through
    bool stop = false; // EOW
    // extra params
};

void add(vt<node>& trie, const string& word) {
    int cur = 0;
    rep(i, 0, sz(word)) {
        int x = word[i] - 'a'; // map to range [0, ALPH - 1]
        if (!trie[cur].down[x]) {
            trie[cur].down[x] = sz(trie);
            trie.pb({});
        }
        cur = trie[cur].down[x];
        trie[cur].cnt++;
    }
    trie[cur].stop = true;
}

int find(const vt<node>& trie, const string& word) {
    int cur = 0;
    rep(i, 0, sz(word)) {
        int x = word[i] - 'a';
        if (!trie[cur].down[x]) {
            return -1;
        }
        cur = trie[cur].down[x];
    }
    if (trie[cur].stop) return cur;
    return -1;
}

void del(vt<node>& trie, const string& word, int cur = 0, int i = 0) {
    if (i == sz(word)) {
        trie[cur].stop = false; // mark EOW as false
        return;
    }

    int x = word[i] - 'a';
    int nxt = trie[cur].down[x];
    if (nxt == 0) return; // word not found

    del(trie, word, nxt, i + 1);
    trie[nxt].cnt--;
    if (trie[nxt].cnt == 0 && !trie[nxt].stop) {
        trie[cur].down[x] = 0;
    }
}

// Fixed-Length Bit Trie (XOR / Prefix Bit Operations)
// - Configured for 30-bit non-negative integers (range [0, 2^30 - 1]).
// - For 64-bit integers (ll): change loop to rrep(i, 61, -1) and bit extraction to (bool)(x & (1LL << i)).
// - Can be used for maximum XOR queries or multiset bit manipulations.
struct bit_node {
    int down[2] = {};
    int cnt = 0;
    // extra params
};

void add(vt<bit_node>& trie, int x) {
    int c = 0;
    rrep(i, 30, -1) {
        trie[c].cnt++;
        bool dir = (bool)(x & (1 << i));
        if (!trie[c].down[dir]) {
            trie[c].down[dir] = sz(trie);
            trie.pb({});
        }
        c = trie[c].down[dir];
    }
    trie[c].cnt++;
}

int find(const vt<bit_node>& trie, int x) {
    int c = 0;
    rrep(i, 30, -1) {
        bool dir = (bool)(x & (1 << i));
        if (trie[c].down[dir]) {
            c = trie[c].down[dir];
        } else {
            c = trie[c].down[!dir];
        }
    }
    return 0; // return result from query
}

void del(vt<bit_node>& trie, int x) {
    int c = 0;
    rrep(i, 30, -1) {
        trie[c].cnt--;
        bool dir = (bool)(x & (1 << i));
        int nx = trie[c].down[dir];
        if (trie[nx].cnt == 1) {
            trie[c].down[dir] = 0;
            return;
        }
        c = nx;
    }
    trie[c].cnt--;
}