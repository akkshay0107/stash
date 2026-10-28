// === utilities ===

template<class T> bool chmin(T& a, const T& b) {
    return b < a ? a = b, 1 : 0;
}

template<class T> bool chmax(T& a, const T& b) {
    return a < b ? a = b, 1 : 0;
}

// lis (use upper_bound for non-decreasing)
int lis(const vi& a) {
    vi dp;
    trav(v, a) {
        int pos = lower_bound(all(dp), v) - dp.begin();
        if (pos == sz(dp)) dp.pb(v);
        else dp[pos] = v;
    }
    return sz(dp);
}

mt19937_64 rng((uint64_t)chrono::steady_clock::now().time_since_epoch().count());

// partial fisher-yates: randomly select k elements
auto fisher_yates = [](vi& popl, vi& res, int k) -> void {
    rep(i, 0, k) {
        uniform_int_distribution<> dist(i, sz(popl) - 1);
        int j = dist(rng);
        swap(popl[i], popl[j]);
    }
    rep(i, 0, k) res[i] = popl[i];
};

// coordinate compression: get_id(x) in [0, sz(vals)); vals[id] restores original value
vi vals = a;
sort(all(vals));
vals.erase(unique(all(vals)), vals.end());
auto get_id = [&](auto x) {
    return lower_bound(all(vals), x) - vals.begin();
};


// === graph traversal ===

// grid bfs
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

int totient[U];
void sieve_phi() {
    iota(totient, totient + U, 0);
    rep(p, 2, U) if (totient[p] == p) {
        for (int j = p; j < U; j += p) {
            totient[j] -= totient[j] / p;
        }
    }
}

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

// inv of a mod m: (x % m + m) % m
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

// unite() returns 1 if merged
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

// 1-indexed; sum(l, r) is [l, r] inclusive
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

    // prefix sum >= k
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

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
OSTree<int> ost;
ost.find_by_order(1);
ost.order_of_key(5);

struct chash {
    const uint64_t C = uint64_t(2e18 * 3.14) + 71;
    const uint32_t RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const {
        return __builtin_bswap64((x ^ RANDOM) * C);
    }
};
gp_hash_table<int, int, chash> ht;

// 0-indexed, [l, r] inclusive
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

// 1-indexed, [l, r] inclusive
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

    // [l, r] inclusive
    ll get_hash(int l, int r) const {
        int w = r - l + 1;
        ll val = p_hash[r + 1] - mod_mul(p_hash[l], pow[w]);
        return (val + M) % M;
    }
};

vll hstring::pow = {1};
const ll hstring::B = uniform_int_distribution<ll>(257, (1LL << 61) - 2)(rng);

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

int max_xor(const vt<bit_node>& trie, int x) {
    int c = 0, res = 0;
    rrep(i, 30, -1) {
        bool dir = (bool)(x & (1 << i));
        if (trie[c].down[!dir] && trie[trie[c].down[!dir]].cnt > 0) {
            res |= (1 << i);
            c = trie[c].down[!dir];
        } else {
            c = trie[c].down[dir];
        }
    }
    return res;
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

// 1-indexed; populates is_bridge[edge_id] and is_cut[u]; handles multigraphs
vi adj_e[U];
pi edges[U];
int tin[U], low[U], timer;
bool is_cut[U], is_bridge[U];

auto dfs_cut = [&](auto&& self, int u, int pe = -1) -> void {
    tin[u] = low[u] = ++timer;
    int ch = 0;
    for (int e : adj_e[u]) {
        if (e == pe) continue;
        int v = edges[e].fr ^ edges[e].se ^ u;
        if (tin[v]) {
            chmin(low[u], tin[v]);
        } else {
            self(self, v, e);
            chmin(low[u], low[v]);
            if (low[v] > tin[u]) is_bridge[e] = true;
            if (low[v] >= tin[u] && pe != -1) is_cut[u] = true;
            ch++;
        }
    }
    if (pe == -1 && ch > 1) is_cut[u] = true;
};
rep(i, 1, n + 1) if (!tin[i]) dfs_cut(dfs_cut, i);

// 2-edge-connected components (contracts bridges using dfs_cut)
int comp_id[U], num_comps;
auto dfs_2cc = [&](auto&& self, int u, int c) -> void {
    comp_id[u] = c;
    for (int e : adj_e[u]) {
        if (is_bridge[e]) continue;
        int v = edges[e].fr ^ edges[e].se ^ u;
        if (!comp_id[v]) self(self, v, c);
    }
};
rep(i, 1, n + 1) if (!comp_id[i]) dfs_2cc(dfs_2cc, i, ++num_comps);

// 1-indexed; scc_id[u] in topological order [0, comp_cnt - 1]
vi adj[U];
int tin[U], low[U], scc_id[U], timer, comp_cnt;
vi st; bool on_st[U];

auto dfs_scc = [&](auto&& self, int u) -> void {
    tin[u] = low[u] = ++timer;
    st.pb(u); on_st[u] = 1;
    for (int v : adj[u]) {
        if (!tin[v]) { self(self, v); chmin(low[u], low[v]); }
        else if (on_st[v]) chmin(low[u], tin[v]);
    }
    if (low[u] == tin[u]) {
        int v;
        do {
            v = st.back(); st.pop_back(); on_st[v] = 0;
            scc_id[v] = comp_cnt;
        } while (v != u);
        comp_cnt++;
    }
};
rep(i, 1, n + 1) if (!tin[i]) dfs_scc(dfs_scc, i);

// eulerian traversal
vi nodes, trail;
vt<char> vis_e;
auto dfs_euler = [&](auto&& self, int u) -> void {
    while (!adj[u].empty()) {
        auto [to, id] = adj[u].back();
        adj[u].pop_back();
        if (vis_e[id]) continue;
        vis_e[id] = 1;
        self(self, to);
        trail.pb(id);
    }
    nodes.pb(u);
};


// === trees ===

// 1-indexed; subtree of u is [tin[u], tout[u]] inclusive
int tin[U], tout[U], timer = 0;
auto euler_tour = [&](auto&& self, int u, int p) -> void {
    tin[u] = ++timer;
    trav(v, adj[u]) if (v != p) self(self, v, u);
    tout[u] = timer;
};
euler_tour(euler_tour, 1, 0);

// centroid decomposition
bool rem[U];
ll st[U];

auto dfs = [&](auto&& self, int u, int p) {
    st[u] = 1;
    trav(v, adj[u]) {
        if(rem[v] || v == p) continue;
        dfs(dfs, v, u);
        st[u] += st[v];
    }
};

auto get_centroid = [&](auto&& get_centroid, int u, int p, int tot) -> int {
    trav(v, adj[u]) {
        if(rem[v] || v == p) continue;
        if(st[v] > tot / 2) return get_centroid(get_centroid, v, u, tot);
    }
    return u;
};

auto dfs_ct = [&](auto&& dfs_ct, int u, int p) -> int {
    dfs(u, -1);
    int c = get_centroid(u, -1, st[u]);
    rem[c] = true;
    // if(p != -1) { cpar[c] = p; ct[p].pb(c); }

    trav(v, adj[c]) {
        if(rem[v]) continue;
        dfs_ct(dfs_ct, v, c);
    }
    return c;
}
int root = dfs_ct(dfs_ct, 1, -1);

// binary lifting / lca
int up[U][20], depth[U];
auto dfs_lca = [&](auto&& self, int u, int p, int d) -> void {
    depth[u] = d; up[u][0] = p;
    rep(i, 1, 20) up[u][i] = up[up[u][i - 1]][i - 1];
    trav(v, adj[u]) if (v != p) self(self, v, u, d + 1);
};
dfs_lca(dfs_lca, 1, 1, 0);

int jmp(int x, int k) {
    rrep(i, 19, -1) if (k & (1 << i)) x = up[x][i];
    return x;
}

int lca(int a, int b) {
    if (depth[a] > depth[b]) swap(a, b);
    b = jmp(b, depth[b] - depth[a]);
    if (a == b) return a;
    rrep(i, 19, -1) {
        if (up[a][i] != up[b][i]) {
            a = up[a][i];
            b = up[b][i];
        }
    }
    return up[a][0];
}

// heavy light decomposition
// tree vertices 1..n; seg tree ranges are 0-indexed [tin[u], tin[u] + sz[u])
int par[U], depth[U], sz[U], heavy[U], head[U], tin[U];

auto dfs_sz = [&](auto&& self, int u, int p, int d) -> void {
    par[u] = p; depth[u] = d; sz[u] = 1; heavy[u] = -1;
    int max_sz = 0;
    trav(v, adj[u]) if (v != p) {
        self(self, v, u, d + 1);
        sz[u] += sz[v];
        if (sz[v] > max_sz) { max_sz = sz[v]; heavy[u] = v; }
    }
};
dfs_sz(dfs_sz, 1, 0, 0);

int timer = 0;
auto dfs_hld = [&](auto&& self, int u, int h) -> void {
    head[u] = h; tin[u] = timer++;
    if (heavy[u] != -1) self(self, heavy[u], h);
    trav(v, adj[u]) {
        if (v != par[u] && v != heavy[u]) self(self, v, v);
    }
};
dfs_hld(dfs_hld, 1, 1);

int lca_hld(int u, int v) {
    for (; head[u] != head[v]; v = par[head[v]]) {
        if (depth[head[u]] > depth[head[v]]) swap(u, v);
    }
    return depth[u] < depth[v] ? u : v;
}

// [l, r) queries
template<class F>
void process_path(int u, int v, F f) {
    for (; head[u] != head[v]; v = par[head[v]]) {
        if (depth[head[u]] > depth[head[v]]) swap(u, v);
        f(tin[head[v]], tin[v] + 1);
    }
    if (depth[u] > depth[v]) swap(u, v);
    f(tin[u], tin[v] + 1); // edge queries: f(tin[u] + 1, tin[v] + 1)
}


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

// 1 = inside, 0 = outside, -1 = boundary
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

// ccw order without collinear points
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
