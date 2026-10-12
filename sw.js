// オフライン対応。画面とデータはネット優先（更新を反映）、ネットがなければ保存分を使う。
// 大きなモデル本体は engine-worker.js が自分でキャッシュするので、ここでは扱わない。
const SHELL = "fishid-shell-v1";
const PRECACHE = ["./", "index.html", "engine-worker.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "apple-touch-icon.png",
                  "data/meta.json", "data/fish.json", "data/lists.json", "data/trivia.json"];
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@";

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET") return;
  const sameOrigin = url.origin === self.location.origin;
  const big = /\/model\/|\/photos\/|\.i8$|\.scale$/.test(url.pathname);
  if (sameOrigin && big) {
    // engine-worker.js が保存したもの（モデル・埋め込み・同梱写真）を優先。
    // ?v=（版）付きは同じ版だけを返す
    e.respondWith(caches.match(req, { ignoreSearch: !url.search }).then(hit => hit || fetch(req)));
    return;
  }
  if (sameOrigin && !big) {
    // ネット優先（ブラウザの HTTP キャッシュも確かめ直して、更新をすぐ反映する）
    e.respondWith(fetch(req.mode === "navigate" ? new Request(req.url, { cache: "no-cache", credentials: "same-origin" })
                                        : new Request(req, { cache: "no-cache" })).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(SHELL).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true })));
  } else if (req.url.startsWith(ORT) || url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com") {
    // バージョン固定のライブラリとフォントは保存分優先
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
      if (res.ok || res.type === "opaque") { const copy = res.clone(); caches.open(SHELL).then(c => c.put(req, copy)); }
      return res;
    })));
  }
});
