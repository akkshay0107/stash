int lis(vi& a){
    vi dp;
    for(int v:a){
        int pos = lower_bound(all(dp),v) - dp.begin();
        if(pos == sz(dp)) dp.push_back(v);
        else dp[pos] = v;
    }
    return sz(dp);
}

// recursive dfs
void dfs(int u){
    vis[u] = true;
    // do some function on u
    for(auto &v : adj[u]){
        if(!vis[v]){
            // do some function on v
            dfs(v);
        }
    }
}

// recursive dfs on trees
void dfs(int u, int p){
    // do some function on u
    for(auto &v : adj[u]){
        if(v != p){
            // do some function on v
            dfs(v,u);
        }
    }
}

//bfs
queue<int> bfs;
while(!bfs.empty()){
    int u = bfs.front(); bfs.pop();
    // do some function on u
    vis[u] = true;
    for(auto &v : adj[u]){
        if(!vis[v]){
            // do some function on v
            bfs.push(v);
        }
    }
}

//flood fill bfs
int dx[4] = {1,-1,0,0};
int dy[4] = {0,0,1,-1};
queue<pi> bfs;
while(!bfs.empty()){
    pi c = bfs.front(); bfs.pop();
    // do some function on u
    rep(k,0,4){
        pi b = {c.fr+dx[k],c.se+dy[k]};
        if(IN(b.fr,-1,n) && IN(b.se,-1,m) && !vis[b.fr][b.se]){
            bfs.push(b);
        }
    }
}

//flood fill dfs same idea adj[u] becomes neighbouring cells

//functional graph
void floyd(){
    // f(i + k*lambda) = f(i)
    // set i = k*lambda => f(2k*lambda) = f(lambda)
    // b -> 2k*lambda and a = k*lambda
    int a=succ(0),b=succ(succ(0));
    while(a!=b){
        a = succ(a);
        b = succ(succ(b));
    }
    a = 0;
    // a -> mu , b -> mu + 2k*lambda
    int mu=0;
    while(a!=b){
        a = succ(a);
        b = succ(b);
        ++mu;
    }
    // a -> mu , b goes from mu + 1 till mu + lambda
    int lambda = 1;
    b = succ(a);
    while(a!=b){
        b = succ(b); ++lambda;
    }
    // start of cycle = mu, cycle length = lambda
}

//prime factorization of numbers spf = smallest prime factor
rep(i,2,U) if(!spf[i]) {
    spf[i] = i;
    for(ll j = (ll)i*i; j < U; j += i) {
        if(!spf[j]) spf[j] = i;
    }
}
rep(i,0,n){
    int temp = a[i];
    while(temp != 1){
        factors[a[i]].pb(spf[temp]);
        temp /= spf[temp];
    }
}

//binary exponentiation a^b
ll exp(ll a,ll b){
    ll res = 1;
    while(b){
        if(b&1){ res = res*a; res %= M;}
        a = a*a; a %= M;
        b >>= 1;
    }
    return res;
}

//modular inverse
ll inv(ll i) {
	return i <= 1 ? i : M - (ll)(M/i) * inv(M % i) % M;
}

// 0-1 bfs
ll dist[U];
d[root] = 0;
deque<int> q;
q.push_front(root);
while(q.size()){
    int c = q.front(); q.pop_front();
    for(auto edge: adj[c]){
        int v = edge.fr; int w = edge.se;
        if(d[v] > d[c]+w){
            d[v] = d[c]+w;
            // if weight is 0 then node comes to front of deque
            // if weight is 1 then node goes to back to deque
            // remains sorted
            if(w) q.push_back(v);
            else q.push_front(v);
        }
    }
}

// union find disjoint set
struct DSU {
	vi e;
    DSU(int N) : e(N,-1) {}
	int get(int x) { return e[x] < 0 ? x : e[x] = get(e[x]); } 
	bool same(int a, int b) { return get(a) == get(b); }
	int size(int x) { return -e[get(x)]; }
	bool unite(int x, int y) {
		x = get(x), y = get(y); if (x == y) return 0;
		if (e[x] > e[y]) swap(x,y);
		e[x] += e[y]; e[y] = x; return 1;
	}
};

// topo sort - dfs
vi ans;
void dfs(int u){
    vis[u] = true;
    for(auto v:adj[u]){
        if(!vis[v]) dfs(v);
    }
    ans.pb(u);
}
void topo_sort(){
    ans.clear(); vis = {};
    rep(i,0,n){
        if(!vis[i]) dfs(i);
    }
    reverse(all(ans));
}

// topo sort - bfs (kahn's algo)
queue<int> q;
//use priority queue for lexicographically minimum order
rep(i,0,n){
    if(indegree[i] == 0) q.push(i);
}
vi order;
while(!q.empty()){
    int c = q.front(); q.pop();
    order.pb(c);
    for(auto v:adj[c]){
        indegree[v]--;
        if(indegree[v]==0) q.push(v);
    }
}
if(sz(order)!=n) // no valid topo sort
else{} // order contains topo sort

// dijkstra
vector<ll> dist(U,LLONG_MAX);
using T = pair<ll,int>;
priority_queue<T,vector<T>,greater<T>> pq;
dist[1] = 0;
pq.push({0,1});
while(!pq.empty()){
    auto c = pq.top(); pq.pop();
    ll cdist = c.fr; int node = c.se;
    if(cdist != dist[node]) continue; // node has been updated before this, old info adds unnecessary iterations
    for(auto v : adj[node]){
        if(dist[v.fr] > cdist + v.se){
            dist[v.fr] = cdist + v.se;
            pq.push({dist[v.fr],v.fr});
        }
    }
}

// floyd warshall
rep(i,1,n+1){
    rep(j,1,n+1) dist[i][j] = INF;
}
rep(i,0,m){
    int a,b; ll c; cin >> a >> b >> c;
    dist[a][b] = min(dist[a][b],c); 
    dist[b][a] = dist[a][b];
}
rep(i,1,n+1) dist[i][i] = 0;
rep(k,1,n+1){
    rep(i,1,n+1){
        rep(j,1,n+1){
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
        }
    }
}

// kruskal mst
vector<pair<ll,pi>> edges;
sort(all(edges));
DSU ds; ds.init(n);
ll cost = 0;
vector<pair<ll,pi>> mst;
for(auto edge: edges){
    if(ds.unite(edge.se.fr,edge.se.se)){
        cost += edge.fr;
        mst.pb({edge.fr,{edge.se.fr,edge.se.se}});
    }
    if(sz(mst)==n-1) break;
}

// prims mst for dense graphs O(V^2)
adj[U][U];
bool vis[U];
vector<pi> min_e(U,{M,-1});
ll cost = 0;
min_e[0].fr = 0; // 0 chosen to be root of mst;
rep(i,0,n){
    int v = -1;
    rep(j,0,n)[
        if(!vis[j]&&(v==-1||min_e[j].fr<min_e[v].fr)) v = j;
    ]
    if(v==-1 || min_e[v].fr == M){
        cost = -1; break; // no mst possible;
    }
    vis[v] = true;
    cost += min_e[v].fr;
    if(min_e[v].se != -1){} // then valid edge of mst formed.
    rep(to,0,n){
        if(adj[v][to]<min_e[to].fr){
            min_e[to] = {adj[v][to],v};
        }
    }
}
// first iteration adds root node for cost 0, rest all start adding edges of tree
// output -> cost
// same idea used for djikstra with dense

// fenwick tree (point update range sum) {one indexed}
// i & (-i) gives us least significant bit
struct FenwickTree{
    vector<ll> bit; int n;
    // O(n) construction
    void init(const vector<ll>& a){
        bit = vector<ll>(sz(a),0);
        n = sz(a);
        rep(i,1,n){
            bit[i] += a[i];
            int r = i + (i & (-i));
            if(r < n) bit[r] += bit[i];
        }
    }
    ll sum(int r){
        ll ret = 0;
        for(;r > 0;r -= r & (-r)) ret += bit[r];
        return ret;
    }
    ll sum(int l,int r){
        return sum(r) - sum(l-1);
    }
    void add(int x,ll delta){
        for(;x < n;x += x & (-x)) bit[x] += delta;
    }
    // check cses/dynrangesum for updates
};

// segment tree (point update range query) {one indexed} [check cses/dynrangemin]
// combine must be associative function [ideally O(1) combination]
// replaced with fast iterative seg tree
node st[4*U];

void build(vi& a, int u, int tl, int tr) {
    if(tl == tr) {
        st[u] = a[tl];
        return;
    }
    int tm = (tl + tr)/2;
    build(a,2*u,tl,tm);
    build(a,2*u+1,tm+1,tr);
    st[u] = combine(st[2*u], st[2*u+1]);
} // build(a,1,1,n)

node get(int l, int r, int u, int tl, int tr) {
    if(l <= tl && tr <= r) return st[u];
    if(l > tr || r < tl) return null; // identity op
    int tm = (tl + tr)/2;
    if(r <= tm) return get(l,r,2*u,tl,tm);
    else if(l > tm) return get(l,r,2*u+1,tm+1,tr);
    else return combine(get(l,r,2*u,tl,tm), get(l,r,2*u+1,tm+1,tr));
} // get(l,r,1,1,n)

void update(int p, int v, int u, int tl, int tr) {
    if(tl == tr) {
        st[u] = v;
        return;
    }
    int tm = (tl + tr)/2;
    if(p <= tm) update(p,v,2*u,tl,tm);
    else update(p,v,2*u+1,tm+1,tr);
    st[u] = combine(st[2*u], st[2*u+1]);
} // update(p,v,1,1,n)

// Order Statistic Tree and Hash table 
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
template <class T>
using OSTree = tree<T,null_type,less<T>,rb_tree_tag,tree_order_statistics_node_update>;
OSTree<int> tree;
tree.find_by_order(1)  // -> iterator for ith element 
tree.order_of_key(5) // -> number of elements lesser than 5
gp_hash_table<int,int,chash> ht; //-> faster hash map
// custom hash for hash table
struct chash {
	const uint64_t C = uint64_t(2e18 * 3.14) + 71;
	const uint32_t RANDOM =
	    chrono::steady_clock::now().time_since_epoch().count();
	size_t operator()(uint64_t x) const {
		return __builtin_bswap64((x ^ RANDOM) * C);
	}
};

// Finding Bridges
int t = 0; 
vector<pi> bridges;
vi tin, low; // low -> lowest back edge ancestor of node in dfs tree
// tin -> entry time
void dfs(int u,int p = 0){
    vis[u] = true;
    tin[u] = low[u] = t++;
    for(int v : adj[u]){
        if(v == p) continue;
        if(vis[v]){
            // (u,v) is a backedge
            low[u] = min(low[u],tin[v]);
        }else{
            dfs(v,u);
            // minimum back edge ancestor of descendant
            low[u] = min(low[u],low[v]);
            // low[v] > tin[u] means only point of entry is (u,v) tree edge
            // on removing tree edge, graph will get disconnected
            if(low[v] > tin[u]){
                bridges.pb({u,v});
            }
        }
    }
}

int find_bridge(){
    t = 0;
    bridges.clear();
    rep(i,1,n+1){
        vis[i] = false; tin[i] = -1; low[i] = -1;
    }
    dfs(1);
    return sz(bridges);
}

// Finding articulation points
int t = 0; 
vi tin, low, art_points; // low -> least depth back edge ancestor of node in dfs tree
// tin -> entry time
void dfs(int u,int p = 0){
    vis[u] = true;
    tin[u] = low[u] = t++;
    int numchild  = 0;
    for(int v:adj[u]){
        if(v == p) continue;
        if(vis[v]){
            // (u,v) is a backedge
            low[u] = min(low[u],tin[v]);
        }else{
            ++numchild;
            dfs(v,u);
            // lowest back edge ancestor from subtree of u
            low[u] = min(low[u],low[v]);
            // low[v] > tin[u] means only point of entry is (u,v) tree edge
            // on removing tree edge, graph will get disconnected
            // low[v] == tin[u] means there exists backedge from subtree of v to u
            // in both cases removal of node u disconnects graph
            if(low[v] >= tin[u] && p != 0){
                art_points.pb(u);
            }
        }
        // in case node is root, numchild > 1 means disconnected graph on removal of root
        if(p == 0 && numchild > 1) art_points.pb(u);
    }
}

int find_art_point(){
    t = 0;
    art_points.clear();
    rep(i,1,n+1){
        vis[i] = false; tin[i] = -1; low[i] = -1;
    }
    dfs(1);
    return sz(art_points);
}

// Euler Tour
// Subtree Queries - check cses/stquery
// LCA - check cses/companyqueries2
// Path Queries - check cses/pathquery

// Euler Totient Sieve
vi get_phi(int n){
    vi phi(n+1);
    iota(all(phi),0);
    rep(i,2,n+1){
        if(phi[i] == i){
            for(int j = i; j <= n; j += i){
                phi[j] -= phi[j]/i;
            }
        }
    }
    return phi;
}

// Euler totient O(sqrt(n))
int phi(int n){
    int ans = n;
    for(int p = 2; p*p <= n; ++p){
        if(n%p == 0){
            while(n%p == 0) n /= p;
            ans -= ans/p;
        }
    }
    if(n > 1) ans -= ans/n;
    return ans;
}

// Extended Euclidean (x and y passed by ref)
int ext_gcd(int a,int b,int& x,int& y){
    if(b == 0){
        x = 1; y = 0;
        return a;
    }
    int xx,yy;
    int g = ext_gcd(b,a%b,xx,yy);
    x = yy;
    y = xx - yy*(a/b);
    return g;
}

// nCr computation in O(n + log M)
ll F[U], iF[U];
F[0] = F[1] = 1;
rep(i,2,U) F[i] = (i*F[i-1])%M;
iF[U-1] = inv(F[U-1]);
rrep(i,U-2,-1) iF[i] = ((i+1)*iF[i+1])%M;

ll C(int n, int r){
    if(!r) return 1;
    else if(r < 0 || r > n) return 0;
    return (F[n]*iF[r]%M)*iF[n-r]%M;
}

// rolling hash for strings
class hstring {
  private:
    // mersenne prime (use 1e9 + 7 for smaller hash)
	static const ll M = (1LL << 61) - 1;
	static const ll B;
	static vll pow;
    // p_hash[i] is the hash of the first i characters of the given string
	vll p_hash;
	__int128 mul(ll a, ll b) { return (__int128)a * b; }
	ll mod_mul(ll a, ll b) { return mul(a, b) % M; }
  public:
	hstring(const string &s) : p_hash(s.size() + 1) {
		while (pow.size() <= s.size()) { pow.push_back(mod_mul(pow.back(), B)); }
		p_hash[0] = 0;
		rep(i,0,sz(s)) p_hash[i + 1] = (mul(p_hash[i], B) + s[i]) % M;
	}
	ll get_hash(int l, int r) {
		int w = (r - l + 1);
        ll val = p_hash[r+1] - mod_mul(p_hash[l],pow[w]);
        return (val + M) % M;
	}
};
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
vll hstring::pow = {1};
const ll hstring::B = uniform_int_distribution<ll>(1, M - 1)(rng);

// sparse table O(nlogn) construction
// O(1) query for idempotent (min,max)
// stores value of function over [i, i + 2^j) for j <= LOG_U;
// check cses/staticmin

// centroid decomposition
// example storing distance to each ancestor in centroid tree
// rem[u] = if u removed from computation of centroid tree
vector<pi> anc[U];

int dfs(int u, int p){
	sz[u] = 1;
	for(auto v: adj[u]){
		if(v!=p && !rem[v]) sz[u] += dfs(v,u);
	}
	return sz[u];
}

int centroid(int u,int p,int n){
	for(auto v:adj[u]){
		if(v==p || rem[v]) continue;
		if(sz[v] > n/2) return centroid(v,u,n);
	}
	return u;
}

void find_dist(int u,int p,int c,int d){
	for(int v: adj[u]){
		if(v==p || rem[v]) continue;
		find_dist(v,u,c,d+1);
	}
	anc[u].pb({c,d});
}

void build_ct(int u,int p){
	int n = dfs(u,p);
	int c = centroid(u,p,n);
	find_dist(c,p,c,0);
	rem[c] = true;
	for (int v: adj[c]){
		if(rem[v]) continue;
		build_ct(v,c);
	}
}

// binary jumping + lca
int l = ceil(log2(n));
vector<vi> up(n+1, vi(l+1,0)); // up[u][k] = 2^k th ancestor of u
vi d(n+1,0); // store all depths in d
// store all parents in up[u][0]
rep(i,1,l+1){
    rep(j,1,n+1){
        up[j][i] = up[up[j][i-1]][i-1];
    }
}
// jumping up k ancestors
auto jmp = [&](int x, int k) -> int {
    if(k > dep[x]) return -1;
    rrep(i,l,-1){
        if(k&(1<<i)){
            x = up[x][i];
        }
    }
    return x;
};

auto lca = [&](int a,int b) -> int {
    if(d[a] > d[b]) swap(a,b);
    b = jmp(b, d[b]-d[a]);
    if(a == b) return a;
    rrep(i,l,-1){
        if(up[a][i] != up[b][i]){
            a = up[a][i];
            b = up[b][i];
        }
    }
    return up[a][0];
};

// fast max/min assignment
void chkmax(int& x, int y) {
    if(x < y) x = y;
}

void chkmin(int& x,int y) {
    int (x > y) x = y;
}

// trie
const int ALPH; // size of alphabet
struct node {
    int down[ALPH] = {};
    int cnt = 0; // reference count
    bool stop = false; // EOW
    // extra params
};
vt<node> trie;

void add(vt<node>& trie, string word) {
    int cur = 0;
    rep(i,0,sz(word)) {
        int x = word[i] - 'a'; // whatever to get it in range [0,ALPH]
        if(!trie[cur].down[x]) {
            trie[cur].down[x] = sz(trie);
            trie.pb({});
        }
        cur = trie[cur].down[x];
        trie[cur].cnt++;
    }
    trie[cur].stop = true;
}

int find(vt<node>& trie, string word) {
    int cur = 0;
    rep(i,0,sz(word)) {
        int x = word[i] - 'a'; // whatever to get it in range [0,ALPH]
        if(!trie[cur].down[x]) {
            return -1
        } else cur = trie[cur].down[x];
    }
    if(cur.stop) return cur;
    return -1;
}

void del(vt<node>& trie, const string& word, int cur = 0, int i = 0) {
    if (i == sz(word)) {
        trie[cur].stop = false; // mark EOW as false
        return;
    }

    int x = word[i] - 'a';
    int nxt = trie[cur].down[x];
    if (nxt == 0) {
        return; // Word not found
    }

    del(trie, word, nxt, i + 1);
    trie[nxt].cnt--;
    if (trie[nxt].cnt == 0 && !trie[nxt].stop) {
        trie[cur].down[x] = 0;
    }
}

// fixed len bit trie
struct node {
    int down[2] = {};
    int cnt = 0;
    // extra params;
};

void add(vt<node>& trie, int x){
    int c = 0;
    rrep(i,30,-1){
        trie[c].cnt++;
        bool dir = (bool) (x & (1<<i));
        if(!trie[c].down[dir]){
            trie[c].down[dir] = sz(trie);
            trie.pb({});
        }
        c = trie[c].down[dir];
    }
    trie[c].cnt++;

}

int find(vt<node>& trie, int x){
    int c = 0;
    rrep(i,30,-1){
        if(x & (1<<i)){
            // do smth
        } else {
            // do smth
        }
    }
    return 0; // return smthing from the extra params
}

void del(vt<node>& trie, int x){
    int c = 0;
    rrep(i,30,-1){
        trie[c].cnt--;
        bool dir = (bool) (x & (1<<i));
        int nx = trie[c].down[dir];
        if(trie[nx].cnt == 1){
            trie[c].down[dir] = 0;
            return;
        }
        c = nx;
    }
    trie[c].cnt--;
}

// Finding 2 edge CCs
// 2CC -> removing any edge from the components keeps it connected
// equivalent to removing all bridges from the graph and checking connected components
vi tin(n+1), low(n+1), comp(n+1);
vvi two_cc;
int timer = 0;
stack<int> st;
auto dfs = [&](auto&& dfs, int u, int p) -> void {
    tin[u] = low[u] = ++timer;
    st.push(u);
    bool multiple_edges = false;

    trav(v, adj[u]) {
        if (v == p && !multiple_edges) {
			multiple_edges = true;
			continue;
		} // multiple edges only for a multigraph otherwise we can remove this section
        if(!tin[v]) {
            dfs(dfs,v,u);
            low[u] = min(low[u], low[v]);
        } else {
            low[u] = min(low[u], tin[v]);
        }
    }
    
    if(tin[u] == low[u]) {
        two_cc.emplace_back();
        while(st.top() != u) {
            two_cc.back().pb(st.top());
            comp[st.top()] = sz(two_cc);
            st.pop();
        }
        two_cc.back().pb(st.top());
        comp[st.top()] = sz(two_cc);
        st.pop();
    }
};

rep(i,1,n+1) {
    if(!comp[i]) {
        dfs(dfs, i, i);
    }
}

// fisher yates partial shuffle
// assumes res is preset to size k
// popl = population vector
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
auto fisher_yates = [&popl](vi& res, int k) -> void {
    rep(i,0,k){
        uniform_int_distribution<> dist(i,sz(popl)-1);
        int j = dist(rng);
        swap(popl[i], popl[j]);
    }
    rep(i,0,k) res[i] = popl[i];
};

// full shuffle
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
vi arr(n);
shuffle(all(arr), rng);

// SCC + condensation graph (Kosaraju)
vt<bool> vis(n+1, false);
auto dfs = [&](auto&& dfs, int u, vvi& adj, vi& out) -> void {
    vis[u] = true;
    trav(v, adj[u]) if(!vis[v]) {
        dfs(dfs,v,adj,out);
    }
    out.pb(u);
};

vvi comps, gscc;
auto scc = [&]() -> void {
    vi order;
    rep(i,1,n+1) if(!vis[i]) dfs(dfs,i,g,order);

    vis.assign(n+1,false);
    reverse(all(order));
    vi roots(n+1,0);
    comps.pb({});

    trav(v,order) {
        if(vis[v]) continue;
        vi cc;
        dfs(dfs,v,gt,cc);
        trav(x,cc) {
            roots[x] = sz(comps);
        }
        comps.pb(cc);
    }

    gscc.assign(sz(comps), {});
    rep(i,1,n+1) {
        int ru = roots[i];
        trav(v, g[i]) {
            int rv = roots[v];
            if(ru != rv) gscc[ru].pb(rv);
        }
    }
    // optional: remove duplicate connections
    rep(i,1,sz(comps))
        sort(all(gscc[i])), gscc[i].erase(unique(all(gscc[i])), gscc[i].end());
};

// check 2SAT in cses/giantpizza