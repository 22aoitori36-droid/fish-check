// 判定エンジン（Web Worker）。モデルとデータの取得・キャッシュ、推論、スコア計算を担当する。
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
importScripts(ORT_URL + "ort.min.js");
ort.env.wasm.wasmPaths = ORT_URL;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const BIG_CACHE = "fishid-big";
let meta, session, fishQ, fishScale, negQ, negScale, protoList, protoAll, lists;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function getMeta() {
  if (!meta) meta = await (await fetch("data/meta.json", { cache: "no-cache" })).json();
  return meta;
}
// 初回にまとめて取得するファイル。URL に内容のハッシュを付け、変わったファイルだけ取り直す
const versioned = (m, path) => `${path}?v=${m.files[path].h}`;
const bigPaths = (m) => Object.keys(m.files);

async function bigCache() {
  // 以前の版（キャッシュ名に版番号を付けていた頃）のものは消す
  for (const k of await caches.keys()) if (k.startsWith("fishid-big-")) await caches.delete(k);
  return caches.open(BIG_CACHE);
}

async function status() {
  const m = await getMeta(), cache = await bigCache();
  const paths = bigPaths(m);
  const hits = await Promise.all(paths.map(p => cache.match(versioned(m, p))));
  const missing = paths.filter((_, i) => !hits[i]);
  const missingBytes = missing.reduce((a, p) => a + m.files[p].n, 0);
  return { cached: !missing.length, total: missingBytes, update: missing.length < paths.length };
}

// キャッシュ優先で取得。なければダウンロードして進捗を送り、キャッシュに入れる
async function fetchAll(m, allowDownload) {
  const cache = await bigCache();
  const paths = bigPaths(m), urls = paths.map(p => versioned(m, p));
  const cached = await Promise.all(urls.map(u => cache.match(u)));
  const missing = paths.filter((_, i) => !cached[i]);
  if (missing.length && !allowDownload) throw new Error("not-cached");
  const total = missing.reduce((a, p) => a + m.files[p].n, 0);
  let loaded = 0, lastPost = 0;
  const out = {};
  async function download(p) {
    const u = versioned(m, p);
    const r = await fetch(u, { cache: "no-store" });
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    const reader = r.body.getReader(), chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.length;
      if (performance.now() - lastPost > 200) { lastPost = performance.now(); post({ type: "progress", loaded, total }); }
    }
    const blob = new Blob(chunks);
    await cache.put(u, new Response(blob, { headers: { "Content-Type": r.headers.get("Content-Type") || "" } }));
    if (!p.startsWith("photos/")) out[p] = await blob.arrayBuffer();
  }
  // 大きいファイルは1つずつ、写真は6本並列
  const photos = missing.filter(p => p.startsWith("photos/")), rest = missing.filter(p => !p.startsWith("photos/"));
  for (const p of rest) await download(p);
  for (let i = 0; i < photos.length; i += 6) await Promise.all(photos.slice(i, i + 6).map(download));
  if (missing.length) post({ type: "progress", loaded: total, total });
  for (let i = 0; i < paths.length; i++)
    if (cached[i] && !paths[i].startsWith("photos/")) out[paths[i]] = await cached[i].arrayBuffer();
  // 使わなくなった古い版を消す
  const keep = new Set(urls.map(u => new URL(u, self.location.href).href));
  for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req);
  return out;
}

async function load(allowDownload) {
  const t0 = performance.now();
  const m = await getMeta();
  const files = await fetchAll(m, allowDownload);
  post({ type: "phase", phase: "init" });
  const model = new Uint8Array(m.model_bytes);
  let off = 0;
  for (const p of m.model_parts) { const b = new Uint8Array(files[p]); model.set(b, off); off += b.length; delete files[p]; }
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  fishQ = new Int8Array(files["data/fish.i8"]); fishScale = new Float32Array(files["data/fish.scale"]);
  negQ = new Int8Array(files["data/neg.i8"]); negScale = new Float32Array(files["data/neg.scale"]);
  const table = (n) => files[`data/${n}.i8`] &&
    { q: new Int8Array(files[`data/${n}.i8`]), scale: new Float32Array(files[`data/${n}.scale`]) };
  protoList = table("proto_list"); protoAll = table("proto_all");
  lists = await (await fetch("data/lists.json")).json();
  post({ type: "ready", ms: Math.round(performance.now() - t0), threads: ort.env.wasm.numThreads });
}

// 量子化された埋め込み行列 q (n×dim) とベクトル v の内積
function dots(q, scale, v) {
  const dim = v.length, n = scale.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * dim;
    for (let k = 0; k < dim; k++) s += q[o + k] * v[k];
    out[i] = s * scale[i] / 127;
  }
  return out;
}

// 複数枚の写真は、それぞれの特徴ベクトルを平均してから正規化する
async function embedAll(pixelsList) {
  const m = meta, v = new Float32Array(m.dim);
  for (const pixels of pixelsList) {
    const input = new ort.Tensor("float32", pixels, [1, 3, m.size, m.size]);
    const e = (await session.run({ pixels: input })).emb.data;
    for (let k = 0; k < m.dim; k++) v[k] += e[k];
  }
  let norm = 0;
  for (let k = 0; k < m.dim; k++) norm += v[k] * v[k];
  norm = Math.sqrt(norm) || 1;
  for (let k = 0; k < m.dim; k++) v[k] /= norm;
  return v;
}

async function identify(pixelsList, listName, k) {
  const m = meta, t0 = performance.now();
  const v = await embedAll(Array.isArray(pixelsList) ? pixelsList : [pixelsList]);
  const tInfer = performance.now() - t0;
  const fs = dots(fishQ, fishScale, v), ns = dots(negQ, negScale, v), L = m.logit_scale;

  // 魚らしさ: 魚 + 非魚類サンプル全体の中で魚に入る確率
  let mx = -Infinity;
  for (const x of fs) mx = Math.max(mx, x);
  for (const x of ns) mx = Math.max(mx, x);
  let pf = 0, pn = 0;
  for (const x of fs) pf += Math.exp(L * (x - mx));
  for (const x of ns) pn += Math.exp(L * (x - mx));

  // 候補（水族館リスト or 全魚類）の中での確率
  const cand = listName && lists[listName] ? lists[listName] : null;
  const idx = cand || Array.from(fs.keys());
  // 展示種は、写真から作った特徴を混ぜたもの（プロトタイプ）で比べる。
  // 指定なしのときは、リスト外の魚が取られすぎないよう展示種のスコアからハンデを引く
  let sim = fs;
  const proto = cand ? protoList : protoAll;
  if (proto) {
    sim = fs.slice();
    const ps = dots(proto.q, proto.scale, v), delta = cand ? 0 : m.proto.delta;
    m.proto_rows.forEach((row, j) => { sim[row] = ps[j] - delta; });
  }
  let cm = -Infinity;
  for (const i of idx) cm = Math.max(cm, sim[i]);
  const probs = new Float32Array(idx.length);
  let z = 0;
  idx.forEach((i, j) => { probs[j] = Math.exp(L * (sim[i] - cm)); z += probs[j]; });
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, k);
  return {
    fish_score: pf / (pf + pn),
    species: order.map(j => ({ row: idx[j], score: probs[j] / z })),
    all: { idx, probs: Array.from(probs, p => p / z) },
    ms: { infer: Math.round(tInfer), total: Math.round(performance.now() - t0) },
  };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "status") post({ type: "status", ...(await status()) });
    else if (data.type === "load") await load(data.allowDownload);
    else if (data.type === "identify") {
      const r = await identify(data.pixels, data.list, data.k || 5);
      post({ type: "result", id: data.id, ...r });
    }
  } catch (e) {
    post({ type: "error", id: data.id, message: String(e && e.message || e) });
  }
};
