// NBA Predictor Service Worker — cache-first for static, network-first for API
const CACHE_NAME = 'nba-predictor-v1';
const STATIC_ASSETS = [
  '/static/style.css',
  '/static/manifest.json',
  '/dashboard',
  '/live',
  '/players',
  '/schedule',
  '/matchups',
  '/allstar',
];

// Install: pre-cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch(() => {
        // Some pages may not exist yet; ignore failures
      });
    })
  );
  self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      );
    })
  );
  self.clients.claim();
});

// Fetch strategy
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests (form submissions, etc.)
  if (event.request.method !== 'GET') return;

  // Skip SSE/streaming endpoints
  if (url.pathname.startsWith('/api/') || url.pathname.includes('/sse')) return;

  // Static assets: cache-first
  if (url.pathname.startsWith('/static/') || url.pathname.startsWith('/images/')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // HTML pages: network-first with cache fallback (for offline)
  if (event.request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => {
          return caches.match(event.request).then((cached) => {
            return cached || caches.match('/dashboard');
          });
        })
    );
    return;
  }
});
