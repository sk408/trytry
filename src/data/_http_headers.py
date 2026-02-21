"""Shared HTTP headers for all outbound requests.

NBA.com and ESPN enforce User-Agent / browser-header checks.  Every module
that hits an external API should import and use these constants so we stay
consistent and easy to update in one place.
"""
from __future__ import annotations

# ---------- stats.nba.com (nba_api stats endpoints) ----------
# The library's built-in headers are missing a User-Agent, which causes
# stats.nba.com to reject requests with connection resets / timeouts.
NBA_STATS_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

# ---------- cdn.nba.com (live scores, schedule JSON) ----------
NBA_CDN_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "cdn.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

# ---------- ESPN / CBS / general web scraping ----------
WEB_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


_patched = False


def patch_nba_api_headers() -> None:
    """Monkey-patch nba_api's global headers so *every* request uses our
    browser-like headers.  Passing ``headers=`` per-endpoint isn't enough
    because the shared ``requests.Session`` caches connection state with
    whatever headers were used first.

    Safe to call multiple times — only patches once.
    Call this once at import time from modules that use nba_api.
    """
    global _patched
    if _patched:
        return
    _patched = True

    # ── Stats API (stats.nba.com) ──
    try:
        from nba_api.stats.library import http as stats_http
        from nba_api.library import http as base_http

        stats_http.STATS_HEADERS = NBA_STATS_HEADERS          # global constant
        stats_http.NBAStatsHTTP.headers = NBA_STATS_HEADERS    # class default
        # Reset any existing session so stale connections are dropped
        stats_http.NBAStatsHTTP._session = None
        base_http.NBAHTTP._session = None
    except Exception:
        pass  # nba_api not installed — nothing to patch

    # ── Live API (cdn.nba.com via nba_api.live) ──
    try:
        from nba_api.live.nba.library import http as live_http

        live_http.NBALiveHTTP.headers = NBA_CDN_HEADERS        # class default
        live_http.NBALiveHTTP._session = None                  # drop stale conn
        # Also patch the module-level STATS_HEADERS if present
        if hasattr(live_http, "STATS_HEADERS"):
            live_http.STATS_HEADERS = NBA_CDN_HEADERS
    except Exception:
        pass  # nba_api live module not available
