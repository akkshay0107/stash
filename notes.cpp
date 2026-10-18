// === utilities ===


// min / max assignment

template<class T> bool chmin(T& a, const T& b) {
    return b < a ? a = b, 1 : 0;
}

template<class T> bool chmax(T& a, const T& b) {
    return a < b ? a = b, 1 : 0;
}


// lis
// strictly increasing, length only. use upper_bound for non-decreasing.
// to recover the subsequence, track tail indices and predecessor indices.

int lis(const vi& a) {
    vi dp;
    trav(v, a) {
        int pos = lower_bound(all(dp), v) - dp.begin();
        if (pos == sz(dp)) dp.pb(v);
        else dp[pos] = v;
    }
    return sz(dp);
}


// randomness
// full shuffle: shuffle(all(arr), rng). use a fixed seed to reproduce a run.

mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());


// partial fisher-yates - randomly select k elements from a population
// sample k positions without replacement in o(k); duplicate values may remain.
// mutates popl; size res to at least k and require 0 <= k <= sz(popl). needs rng.

auto fisher_yates = [](vi& popl, vi& res, int k) -> void {
    rep(i, 0, k) {
        uniform_int_distribution<> dist(i, sz(popl) - 1);
        int j = dist(rng);
        swap(popl[i], popl[j]);
    }
    rep(i, 0, k) res[i] = popl[i];
};


// coordinate compression
// use ranks as array/tree indices when values are large or sparse.
// include all values that need exact ranks; absent x gives an insertion position.
// ranks preserve order, not distance; use vals[id] to recover the original value.

vi vals = a;
sort(all(vals));
vals.erase(unique(all(vals)), vals.end());
auto get_id = [&](auto x) {
    return lower_bound(all(vals), x) - vals.begin();
};


// === graph traversal ===


// grid bfs
// seed and mark the source before the loop; seed all sources for multi-source bfs.
// add wall/valid-cell checks to the neighbor condition; change dx/dy for diagonals.
// for distances or path recovery, record distance/parent when enqueuing. 0-indexed.

int dx[4] = {1, -1, 0, 0};
int dy[4] = {0, 0, 1, -1};
queue<pi> bfs;
while (!bfs.empty()) {
    pi c = bfs.front(); bfs.pop();
    rep(k, 0, 4) {
        int xx = c.fr + dx[k];
        int yy = c.se + dy[k];
        if (IN(xx, 0, n - 1) && IN(yy, 0, m - 1) && !vis[xx][yy]) {
            vis[xx][yy] = true;
            bfs.push({xx, yy});
        }
    }
}


// floyd cycle detection
// find a cycle in a successor sequence using constant extra space; define succ(u).
// return mu (steps before entry), lambda (cycle length), and/or a (entry vertex).
// succ must exist at every reachable vertex; null-terminated lists need end checks.

void floyd(int start = 0) {
    int a = succ(start);
    int b = succ(succ(start));
    while (a != b) {
        a = succ(a);
        b = succ(succ(b));
    }
    a = start;
    int mu = 0;
    while (a != b) {
        a = succ(a);
        b = succ(b);
        ++mu;
    }
    int lambda = 1;
    b = succ(a);
    while (a != b) {
        b = succ(b);
        ++lambda;
    }
}


// topological sort / dfs
// order vertices before their dependents, then process order for dag dp.
// 1-indexed; assumes a dag. use kahn if cycles need detecting.

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
    reverse(all(order));
}


// topological sort / kahn
// compute indegrees first; this consumes them. 1-indexed; use [0, n) for 0-indexing.
// sz(order) != n means a cycle; unprocessed vertices are on cycles or downstream.
// use a min-heap for the lexicographically smallest order; use order for dag dp.

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

// === number theory ===


// factorization / smallest prime factors
// call sieve_spf(), then factorize(x) for 1 <= x < U; returns {prime, exponent}.
// factorize(1) is empty. for x >= 2, spf[x] == x also tests primality.

int spf[U];

void sieve_spf() {
    fill(spf, spf + U, 0);
    rep(i, 2, U) if (!spf[i]) {
        spf[i] = i;
        for (ll j = 1LL * i * i; j < U; j += i) {
            if (!spf[j]) spf[j] = i;
        }
    }
}

vt<pi> factorize(int x) {
    vt<pi> factors;
    while (x > 1) {
        int p = spf[x], count = 0;
        while (x % p == 0) {
            x /= p;
            ++count;
        }
        factors.pb({p, count});
    }
    return factors;
}


// mobius sieve
// call sieve_mu(), then mu[x] for x < U; use for divisor inclusion-exclusion.
// mu[1] = 1; mu[x] is zero for squared prime factors, otherwise (-1)^k
// for k distinct primes.

int mu[U];

void sieve_mu() {
    fill(mu, mu + U, 1);
    vt<bool> composite(U, false);
    mu[0] = 0;
    rep(p, 2, U) if (!composite[p]) {
        for (int j = p; j < U; j += p) {
            composite[j] = true;
            mu[j] = -mu[j];
        }
        ll square = 1LL * p * p;
        for (ll j = square; j < U; j += square) mu[j] = 0;
    }
}


// totient / single value
// count integers in [1, n] coprime to n; n >= 1. no precomputation needed.

int phi(int n) {
    int ans = n;
    if (n % 2 == 0) {
        while (n % 2 == 0) n /= 2;
        ans -= ans / 2;
    }
    for (int p = 3; (ll)p * p <= n; p += 2) {
        if (n % p == 0) {
            while (n % p == 0) n /= p;
            ans -= ans / p;
        }
    }
    if (n > 1) ans -= ans / n;
    return ans;
}


// totient / table
// call sieve_phi(), then totient[x] for x < U; totient[0] = 0, totient[1] = 1.
// use phi(n) above for a single value without precomputation.

int totient[U];

void sieve_phi() {
    iota(totient, totient + U, 0);
    rep(p, 2, U) if (totient[p] == p) {
        for (int j = p; j < U; j += p) {
            totient[j] -= totient[j] / p;
        }
    }
}


// binary exponentiation
// a^b modulo M, including negative a; require b >= 0 and M > 0.
// widen multiplication to __int128 if products of residues can exceed ll.

ll exp(ll a, ll b) {
    a %= M;
    if (a < 0) a += M;
    ll res = 1 % M;
    while (b) {
        if (b & 1) res = res * a % M;
        b >>= 1;
        if (b) a = a * a % M;
    }
    return res;
}


// extended gcd
// ext_gcd(a, b, x, y) returns g with a*x + b*y = g; a, b >= 0, not both zero.
// for inverse of a mod m, require g = 1 and normalize x as (x % m + m) % m.
// for a*x + b*y = c, solutions exist iff g divides c; scale x,y by c/g.

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


// prime modulus / inverse + combinations
// inv(a) requires 1 <= a < M and prime M; use ext_gcd for composite moduli.
// for C(n, r), call precompute_comb(limit), 0 <= limit < min(U, M); 0 <= n <= limit.
// lucas(n, r) is the optional large-n extension: needs M <= U and the full
// precompute_comb(M - 1). all modular products must fit ll.

ll inv(ll i) {
    return i <= 1 ? i : M - (ll)(M / i) * inv(M % i) % M;
}

ll F[U], iF[U];
void precompute_comb(int limit) {
    F[0] = 1;
    rep(i, 1, limit + 1) F[i] = F[i - 1] * i % M;
    iF[limit] = inv(F[limit]);
    rrep(i, limit, 0) iF[i - 1] = iF[i] * i % M;
}

ll C(int n, int r) {
    if (r < 0 || r > n) return 0;
    return F[n] * iF[r] % M * iF[n - r] % M;
}

ll lucas(ll n, ll r) {
    if (r < 0 || r > n) return 0;
    ll res = 1;
    while (n) {
        res = res * C(n % M, r % M) % M;
        n /= M;
        r /= M;
    }
    return res;
}


// matrix exponentiation
// use for small-state linear recurrences or counting walks of exactly p edges.
// define the state and transition matrix, raise it to p, then apply to the initial
// state. adjacency powers give walk counts; o(n^3 log p) total.

struct Matrix {
    int n;
    vvl mat;
    Matrix(int n) : n(n), mat(n, vll(n, 0)) {}
    static Matrix identity(int n) {
        Matrix res(n);
        rep(i, 0, n) res.mat[i][i] = 1;
        return res;
    }
    Matrix operator*(const Matrix& o) const {
        Matrix res(n);
        rep(i, 0, n) {
            rep(k, 0, n) {
                if (!mat[i][k]) continue;
                rep(j, 0, n) {
                    res.mat[i][j] = (res.mat[i][j] + mat[i][k] * o.mat[k][j]) % M;
                }
            }
        }
        return res;
    }
};

Matrix mat_exp(Matrix a, ll p) {
    Matrix res = Matrix::identity(a.n);
    while (p) {
        if (p & 1) res = res * a;
        p >>= 1;
        if (p) a = a * a;
    }
    return res;
}


// === shortest paths ===


// 0-1 bfs
// shortest paths with weights only 0 or 1; adj[u] stores {v, w}, source root.
// seed every source at distance zero for multi-source queries. add parent[v]
// on successful relaxation to recover paths; do not mark visited on discovery.

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


// dijkstra
// nonnegative edge weights; adj[u] stores {v, w}. change source 1 in both places,
// or seed several sources at zero. store parent[v] on relaxation to recover paths.
// unreachable distances are LLONG_MAX; finite path sums must fit ll.

vll dist(n + 1, LLONG_MAX);
using T = pair<ll, int>;
priority_queue<T, vt<T>, greater<T>> pq;
dist[1] = 0;
pq.push({0, 1});
while (!pq.empty()) {
    auto [cdist, node] = pq.top(); pq.pop();
    if (cdist != dist[node]) continue;
    trav(edge, adj[node]) {
        auto [v, w] = edge;
        if (dist[v] > cdist + w) {
            dist[v] = cdist + w;
            pq.push({dist[v], v});
        }
    }
}


// floyd-warshall
// all-pairs shortest paths for small n; o(n^3). directed input; also relax the
// reverse edge for undirected input. dist[i][i] < 0 detects a negative cycle.
// for path recovery store the next hop on improvement. keep k outermost and
// choose INF with headroom for adding finite distances.

rep(i, 1, n + 1) {
    rep(j, 1, n + 1) dist[i][j] = INF;
    dist[i][i] = 0;
}
rep(i, 0, m) {
    int a, b; ll c; cin >> a >> b >> c;
    chmin(dist[a][b], c);
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


// === dsu + spanning trees ===


// dsu
// maintain connectivity under merges; use DSU(n + 1) for vertices 1..n.
// unite reports a successful merge. add component sums/minima/etc. in unite;
// the chosen representative is not necessarily the smallest vertex.

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


// kruskal
// minimum spanning tree of an undirected graph; fill edges, then sort(all(edges)).
// reverse the weight order for a maximum spanning tree. fewer than n-1 selected
// edges means disconnected; mst then holds a minimum spanning forest. needs dsu.

struct Edge {
    int u, v;
    ll w;
    bool operator<(const Edge& o) const { return w < o.w; }
};

vt<Edge> edges;
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


// prim / dense graph
// mst with an adjacency matrix, o(n^2); 0-indexed. initialize missing edges to M
// and keep real weights below M. collect {v, min_e[v].se} in the empty branch.
// cost = -1 signals disconnection; this can also be a valid cost with negative weights.

const int MAXV = 2500;
int adj[MAXV][MAXV];
bool vis[MAXV];
vt<pi> min_e(MAXV, {M, -1});
ll cost = 0;
min_e[0].fr = 0;
rep(i, 0, n) {
    int v = -1;
    rep(j, 0, n) {
        if (!vis[j] && (v == -1 || min_e[j].fr < min_e[v].fr)) v = j;
    }
    if (v == -1 || min_e[v].fr == M) {
        cost = -1; break;
    }
    vis[v] = true;
    cost += min_e[v].fr;
    if (min_e[v].se != -1) {
    }
    rep(to, 0, n) {
        if (adj[v][to] < min_e[to].fr) {
            min_e[to] = {adj[v][to], v};
        }
    }
}


// === range queries ===


// fenwick tree
// point add + inclusive range sum, 1-indexed; array constructor needs dummy a[0].
// for assignment, add new-old. for range add + point query, store a difference
// array; range add + range sum needs two trees. never call add at index zero.
// lower_bound(k) finds the kth item using nonnegative frequencies, k > 0;
// returns n+1 when total < k.

struct FenwickTree {
    int n;
    vt<ll> bit;

    FenwickTree() : n(0), bit(1, 0) {}
    FenwickTree(int n) : n(n), bit(n + 1, 0) {}
    FenwickTree(const vt<ll>& a) : n(sz(a) - 1), bit(a) {
        bit[0] = 0;
        for (int i = 1; i <= n; i++) {
            int r = i + (i & -i);
            if (r <= n) bit[r] += bit[i];
        }
    }

    ll sum(int r) const {
        ll ret = 0;
        for (; r > 0; r &= r - 1) ret += bit[r];
        return ret;
    }

    ll sum(int l, int r) const { return sum(r) - sum(l - 1); }

    void add(int x, ll delta) {
        for (; x <= n; x += x & -x) bit[x] += delta;
    }

    int lower_bound(ll k) const {
        int pos = 0;
        for (int pw = n ? 1 << __lg(n) : 0; pw; pw >>= 1) {
            if (pos + pw <= n && bit[pos + pw] < k) {
                pos += pw;
                k -= bit[pos];
            }
        }
        return pos + 1;
    }
};


// iterative segment tree
// point assignment + queries [l, r), 0-indexed. define node, combine, and its
// default identity: sum/0, min/inf, max/-inf; richer queries need richer nodes.
// walk finds the first value >= x using range maxima; pad n to a power of two
// and change the branch test when adapting the search to another aggregate.

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

node get(int l, int r) {
    node resl, resr;
    l += n; r += n;
    for (; l < r; l >>= 1, r >>= 1) {
        if (l & 1) resl = combine(resl, seg[l++]);
        if (r & 1) resr = combine(seg[--r], resr);
    }
    return combine(resl, resr);
}

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


// order statistic tree
// find_by_order(k): iterator to zero-based kth key; order_of_key(x): count < x.
// keys are unique; use {value, unique_id} for duplicates. OSTree is in template.cpp.

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
OSTree<int> ost;
ost.find_by_order(1);
ost.order_of_key(5);


// custom hash table
// integer-key map with randomized hashing for adversarial contest inputs.
// change the key/value types as needed; needs the pb_ds include and namespace.

struct chash {
    const uint64_t C = uint64_t(2e18 * 3.14) + 71;
    const uint32_t RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const {
        return __builtin_bswap64((x ^ RANDOM) * C);
    }
};
gp_hash_table<int, int, chash> ht;


// sparse table / range minimum
// static array, inclusive queries [l, r], 0-indexed; o(n log n) build, o(1) query.
// replace min in both build and query for max/gcd/and/or. the operation must be
// associative and idempotent; this overlapping-block query does not support sum.

struct SparseTable {
    int n, K;
    vvi st;
    SparseTable(const vi& a) : n(sz(a)), K(__lg(max(1, n)) + 1), st(K, vi(n)) {
        st[0] = a;
        rep(i, 1, K) {
            rep(j, 0, n - (1 << i) + 1) {
                st[i][j] = min(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
            }
        }
    }
    int query(int l, int r) const {
        int i = __lg(r - l + 1);
        return min(st[i][l], st[i][r - (1 << i) + 1]);
    }
};


// lazy segment tree / range add + minimum
// inclusive [1, n], range add + range min; clear lazy before building/reusing.
// for range sum: change every min merge to +, query identity to 0, and prop to
// add len*fa; widen types if needed. for max, use max and -inf instead.
// for range assignment: overwrite tags and track whether one exists separately;
// set seg[u] to the value (min/max) or len*value (sum). zero is a valid assignment.
// 1e9 must bound all minima in the current version.
// walk(1, 1, n) finds the first negative value, returning n+1 when absent.

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
    seg[u] += fa;
    lazy[u] += fa;
}

inline void push(int u, int tl, int tr) {
    if (lazy[u] == 0) return;
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
    if (l > tr || r < tl) return 1e9;
    if (l <= tl && tr <= r) return seg[u];

    push(u, tl, tr);
    int tm = (tl + tr) / 2;
    int left = get(l, r, 2 * u, tl, tm);
    int right = get(l, r, 2 * u + 1, tm + 1, tr);
    return min(left, right);
}

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


// === strings ===


// polynomial hash
// compare substrings with inclusive get_hash(l, r), 0-indexed; compare lengths too.
// needs rng. equal hashes can collide; use a second independent hash or verify
// matches when necessary. for palindrome queries, also hash the reversed string.

class hstring {
private:
    static const ll M = (1LL << 61) - 1;
    static const ll B;
    static vll pow;
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


// prefix trie
// start with one empty root. lowercase a..z; change ALPH and character mapping
// together for another alphabet. find returns terminal id or -1.
// for nonempty prefix counts, use the reached node's cnt without requiring stop.
// duplicates need a terminal count instead of bool stop; adjust deletion too.
// delete only existing words; detached nodes are not reclaimed.

const int ALPH = 26;
struct node {
    int down[ALPH] = {};
    int cnt = 0;
    bool stop = false;
};

void add(vt<node>& trie, const string& word) {
    int cur = 0;
    rep(i, 0, sz(word)) {
        int x = word[i] - 'a';
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
        trie[cur].stop = false;
        return;
    }

    int x = word[i] - 'a';
    int nxt = trie[cur].down[x];
    if (nxt == 0) return;

    del(trie, word, nxt, i + 1);
    trie[nxt].cnt--;
    if (trie[nxt].cnt == 0 && !trie[nxt].stop) {
        trie[cur].down[x] = 0;
    }
}


// bit trie
// start with one empty root; stores nonnegative 31-bit ints, including duplicates.
// finish find by accumulating xor: prefer equal bits for min xor, opposite for max.
// for kth queries, branch by subtree cnt instead. wider keys need wider parameters,
// masks, and loop bounds together. query nonempty tries; delete existing values.

struct bit_node {
    int down[2] = {};
    int cnt = 0;
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
    return 0;
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


// === connectivity ===


// bridges
// edges whose removal disconnects an undirected component; vertices 1..n.
// parallel edges need incoming edge ids instead of skipping every v == p.
// use TwoCC to contract the graph into a bridge forest.

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
                    chmin(low[u], tin[v]);
                } else {
                    self(self, v, u);
                    chmin(low[u], low[v]);
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


// articulation points
// vertices whose removal disconnects an undirected component; vertices 1..n.
// art_points lists each once. counting the resulting pieces needs child counts
// at low[v] >= tin[u], with separate handling for dfs roots.

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


// 2-edge-connected components
// contract regions with no bridges for tree/path queries about critical edges.
// undirected, 1-indexed; parallel edges allowed. comp_id maps to 1..num_comps;
// call build_tree() for the bridge tree, or forest if disconnected.

struct TwoCC {
    struct DSU {
        vi e, h;
        const vi& depth;

        DSU(int N, const vi& d) : e(N, -1), h(N), depth(d) {
            iota(all(h), 0);
        }

        int get(int x) { return e[x] < 0 ? x : e[x] = get(e[x]); }
        bool same(int a, int b) { return get(a) == get(b); }
        int size(int x) { return -e[get(x)]; }
        int highest(int x) { return h[get(x)]; }

        bool unite(int x, int y) {
            x = get(x); y = get(y);
            if (x == y) return 0;

            int minh = (depth[h[x]] < depth[h[y]]) ? h[x] : h[y];

            if (e[x] > e[y]) swap(x, y);
            e[x] += e[y];
            e[y] = x;

            h[x] = minh;
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


// strongly connected components / kosaraju
// compress directed cycles into a dag for reachability/dp; vertices 1..n.
// comp_id maps vertices to comps; ids follow topological order, and dag is
// deduplicated. aggregate vertex weights per component before dag dp.
// for 2-sat, a variable and its negation sharing an scc means impossible;
// see cses/giantpizza. uses c++20 ranges.

struct SCC {
    int n;
    vi comp_id;
    vvi comps;
    vvi dag;

    SCC(const vvi& g) : n(sz(g) - 1), comp_id(n + 1, 0) {
        vvi gt(n + 1);
        rep(u, 1, n + 1) {
            trav(v, g[u]) gt[v].pb(u);
        }

        vt<char> vis(n + 1, 0);
        vi order; order.reserve(n);

        auto dfs1 = [&](auto&& self, int u) -> void {
            vis[u] = 1;
            trav(v, g[u]) if (!vis[v]) self(self, v);
            order.pb(u);
        };

        rep(i, 1, n + 1) if (!vis[i]) dfs1(dfs1, i);

        vis.assign(n + 1, 0);
        comps.pb({});

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


// eulerian traversal / hierholzer
// use every edge once; adj[u] stores {to, id}, vis is sized for all edge ids.
// undirected copies share an id; adjacency is consumed. replace the outer loop
// with one dfs from the chosen start, then reverse and check sz(edges) == m.
// cycle: even undirected degrees, or equal directed in/out degrees. open trail:
// start at an odd vertex (exactly two), or at out-in = 1 (one matching end,
// all others balanced). all non-isolated vertices must be weakly connected.

vi nodes, edges;
vt<char> vis;
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
}


// === trees ===


// euler tour / subtree flattening
// turn subtree queries into range queries on [tin[u], tout[u]], inclusive.
// store vertex values at tin[u] in a fenwick/segment tree. size tin/tout to n+1,
// reset timer, then euler_tour(root, 0, adj); vertices 1..n.

vi tin, tout;
int timer = 0;

void euler_tour(int u, int p, const vvi& adj) {
    tin[u] = ++timer;
    trav(v, adj[u]) {
        if (v != p) euler_tour(v, u, adj);
    }
    tout[u] = timer;
}


// centroid decomposition
// for repeated distance-to-marked-node queries, maintain best[c] over centroid
// ancestors; update along anc[u], query min(best[c] + distance(u, c)).
// for counting paths, seed distance zero for the centroid, then query each child
// against earlier children before adding its data; do this before marking removed.
// 1-indexed, unweighted; weighted distances need edge weights and wider anc entries.
// clear anc/rem before rebuilding.

vt<pi> anc[U];
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


// binary lifting / lca
// LCA(n, root, adj) on a 1-indexed tree. jmp(u, k) returns -1 above the root;
// dist counts edges. for weighted distances, keep d as depth and add a separate
// root-distance array. for path min/max/sum, add an aggregate alongside each jump.

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


// === geometry 3d ===

template<class T>
struct pt3 {
    T x{}, y{}, z{};
    pt3& operator+=(const pt3& p) { x += p.x; y += p.y; z += p.z; return *this; }
    pt3& operator-=(const pt3& p) { x -= p.x; y -= p.y; z -= p.z; return *this; }
    pt3& operator*=(T k) { x *= k; y *= k; z *= k; return *this; }
    pt3& operator/=(T k) { x /= k; y /= k; z /= k; return *this; }
    pt3 operator+(const pt3& p) const { return {x + p.x, y + p.y, z + p.z}; }
    pt3 operator-(const pt3& p) const { return {x - p.x, y - p.y, z - p.z}; }
    pt3 operator-() const { return {-x, -y, -z}; }
    pt3 operator*(T k) const { return {x * k, y * k, z * k}; }
    pt3 operator/(T k) const { return {x / k, y / k, z / k}; }
    friend pt3 operator*(T k, const pt3& p) { return p * k; }
    bool operator==(const pt3& p) const { return x == p.x && y == p.y && z == p.z; }
};

template<class T>
T dot(pt3<T> a, pt3<T> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<class T>
pt3<T> cross(pt3<T> a, pt3<T> b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

template<class T>
T triple(pt3<T> a, pt3<T> b, pt3<T> c) {
    return dot(a, cross(b, c));
}

template<class T>
T norm(pt3<T> a) { return dot(a, a); }

template<class T>
ld abs(pt3<T> a) {
    ld x = a.x, y = a.y, z = a.z;
    return sqrtl(x * x + y * y + z * z);
}

template<class T>
ld dist(pt3<T> a, pt3<T> b) {
    return abs(pt3<ld>{(ld)a.x - (ld)b.x, (ld)a.y - (ld)b.y, (ld)a.z - (ld)b.z});
}

template<class T>
ld proj(pt3<T> a, pt3<T> b) {
    pt3<ld> u{(ld)a.x, (ld)a.y, (ld)a.z}, v{(ld)b.x, (ld)b.y, (ld)b.z};
    return dot(u, v) / abs(v);
}

template<class T>
ld angle(pt3<T> a, pt3<T> b) {
    pt3<ld> u{(ld)a.x, (ld)a.y, (ld)a.z}, v{(ld)b.x, (ld)b.y, (ld)b.z};
    return acosl(clamp(dot(u, v) / abs(u) / abs(v), -1.0L, 1.0L));
}

// === geometry 2d ===

template<class T>
struct pt2 {
    T x{}, y{};
    pt2& operator+=(const pt2& p) { x += p.x; y += p.y; return *this; }
    pt2& operator-=(const pt2& p) { x -= p.x; y -= p.y; return *this; }
    pt2& operator*=(T k) { x *= k; y *= k; return *this; }
    pt2& operator/=(T k) { x /= k; y /= k; return *this; }
    pt2 operator+(const pt2& p) const { return {x + p.x, y + p.y}; }
    pt2 operator-(const pt2& p) const { return {x - p.x, y - p.y}; }
    pt2 operator-() const { return {-x, -y}; }
    pt2 operator*(T k) const { return {x * k, y * k}; }
    pt2 operator/(T k) const { return {x / k, y / k}; }
    friend pt2 operator*(T k, const pt2& p) { return p * k; }
    bool operator==(const pt2& p) const { return x == p.x && y == p.y; }
    bool operator<(const pt2& p) const { return x < p.x || (x == p.x && y < p.y); }
};

template<class T>
T dot(pt2<T> a, pt2<T> b) {
    return a.x * b.x + a.y * b.y;
}

template<class T>
T cross(pt2<T> a, pt2<T> b) {
    return a.x * b.y - a.y * b.x;
}

template<class T>
T cross(pt2<T> a, pt2<T> b, pt2<T> c) {
    return cross(b - a, c - a);
}

template<class T>
T norm(pt2<T> a) { return dot(a, a); }

template<class T>
ld abs(pt2<T> a) {
    ld x = a.x, y = a.y;
    return sqrtl(x * x + y * y);
}

template<class T>
ld dist(pt2<T> a, pt2<T> b) {
    return abs(pt2<ld>{(ld)a.x - (ld)b.x, (ld)a.y - (ld)b.y});
}

template<class T>
ld proj(pt2<T> a, pt2<T> b) {
    pt2<ld> u{(ld)a.x, (ld)a.y}, v{(ld)b.x, (ld)b.y};
    return dot(u, v) / abs(v);
}

template<class T>
ld angle(pt2<T> a, pt2<T> b) {
    pt2<ld> u{(ld)a.x, (ld)a.y}, v{(ld)b.x, (ld)b.y};
    return acosl(clamp(dot(u, v) / abs(u) / abs(v), -1.0L, 1.0L));
}

template<class T>
struct line2 {
    pt2<T> a, b;
    bool operator||(const line2& l) const { return cross(b - a, l.b - l.a) == 0; }
    bool operator==(const line2& l) const { return (*this || l) && cross(a, b, l.a) == 0; }
};

template<class T>
bool on_line(pt2<T> p, line2<T> l) {
    return cross(l.a, l.b, p) == 0;
}

template<class T>
bool on_segment(pt2<T> p, line2<T> s) {
    return cross(s.a, s.b, p) == 0 &&
           min(s.a.x, s.b.x) <= p.x && p.x <= max(s.a.x, s.b.x) &&
           min(s.a.y, s.b.y) <= p.y && p.y <= max(s.a.y, s.b.y);
}

template<class T>
pt2<ld> line_intersection(line2<T> l, line2<T> m) {
    pt2<T> u = l.b - l.a, v = m.b - m.a;
    ld t = (ld)cross(m.a - l.a, v) / (ld)cross(u, v);
    return {(ld)l.a.x + t * (ld)u.x, (ld)l.a.y + t * (ld)u.y};
}

template<class T>
bool segments_intersect(line2<T> s, line2<T> t) {
    T a = cross(s.a, s.b, t.a), b = cross(s.a, s.b, t.b);
    T c = cross(t.a, t.b, s.a), d = cross(t.a, t.b, s.b);

    if (a == 0 && on_segment(t.a, s)) return true;
    if (b == 0 && on_segment(t.b, s)) return true;
    if (c == 0 && on_segment(s.a, t)) return true;
    if (d == 0 && on_segment(s.b, t)) return true;
    
    return ((a < 0 && b > 0) || (a > 0 && b < 0)) &&
           ((c < 0 && d > 0) || (c > 0 && d < 0));
}

template<class T>
pt2<ld> project(pt2<T> p, line2<T> l) {
    pt2<ld> a{(ld)l.a.x, (ld)l.a.y}, b{(ld)l.b.x, (ld)l.b.y};
    pt2<ld> q{(ld)p.x, (ld)p.y}, d = b - a;
    return a + d * (dot(q - a, d) / norm(d));
}

template<class T>
pt2<ld> closest(pt2<T> p, line2<T> s) {
    pt2<ld> a{(ld)s.a.x, (ld)s.a.y}, b{(ld)s.b.x, (ld)s.b.y};
    if (s.a == s.b) return a;
    
    pt2<ld> q{(ld)p.x, (ld)p.y}, d = b - a;
    return a + d * clamp(dot(q - a, d) / norm(d), 0.0L, 1.0L);
}

template<class T>
ld dist_line(pt2<T> p, line2<T> l) {
    return dist(pt2<ld>{(ld)p.x, (ld)p.y}, project(p, l));
}

template<class T>
ld dist_segment(pt2<T> p, line2<T> s) {
    return dist(pt2<ld>{(ld)p.x, (ld)p.y}, closest(p, s));
}

// === polygons + convex hull ===

template<class T>
T signed_area2(const vt<pt2<T>>& p) {
    T res = 0;
    int n = (int)p.size();
    rep(i, 0, n) res += cross(p[i], p[(i + 1) % n]);
    return res;
}

template<class T>
ld area(const vt<pt2<T>>& p) {
    return fabsl((ld)signed_area2(p)) / 2.0L;
}

template<class T>
ld perimeter(const vt<pt2<T>>& p) {
    ld res = 0;
    int n = (int)p.size();
    rep(i, 0, n) res += dist(p[i], p[(i + 1) % n]);
    return res;
}

template<class T>
int point_in_polygon(const vt<pt2<T>>& p, pt2<T> q) {
    bool inside = false;
    int n = (int)p.size();
    rep(i, 0, n) {
        pt2<T> a = p[i], b = p[(i + 1) % n];
        if (on_segment(q, line2<T>{a, b})) return -1;
        if (a.y > b.y) swap(a, b);
        if (a.y <= q.y && q.y < b.y && cross(a, b, q) > 0) inside = !inside;
    }
    return (int)inside;
}

template<class T>
vt<pt2<T>> convex_hull(vt<pt2<T>> p) {
    sort(all(p));
    p.erase(unique(all(p)), p.end());
    int n = (int)p.size();
    if (n <= 2) return p;
    vt<pt2<T>> h;
    trav(q, p) {
        while (h.size() >= 2 && cross(h[h.size() - 2], h.back(), q) <= 0) h.pop_back();
        h.pb(q);
    }
    int lower = (int)h.size();
    for (int i = n - 2; i >= 0; --i) {
        while ((int)h.size() > lower && cross(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
        h.pb(p[i]);
    }
    h.pop_back();
    return h;
}
