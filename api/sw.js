const CACHE = 'autoshot-v1';
const SHELL  = ['/', '/app', '/manifest.json', '/icon.svg', '/icon-maskable.svg'];

// Install — cache app shell
self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
});

// Activate — clean old caches
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch — network-first for API, cache-first for shell
self.addEventListener('fetch', e => {
  const url = e.request.url;

  // Always network for API endpoints
  if (['/identify', '/detect', '/analyze', '/health'].some(p => url.includes(p))) return;

  // Cache-first for everything else
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request).then(res => {
      // Cache new shell resources on the fly
      if (res.ok && e.request.method === 'GET') {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return res;
    }))
  );
});
