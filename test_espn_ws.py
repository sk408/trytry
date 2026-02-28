"""Test ESPN WebSocket for real-time game updates.

Uses ESPN's Fastcast service (same approach as the Linedrive library)
to get real-time play-by-play, score updates, and game events via WebSocket.

Usage: python test_espn_ws.py [team_name]
  e.g. python test_espn_ws.py nuggets
  If no team specified, lists all live games.
"""

import json
import sys
import time
import zlib
import base64
from datetime import datetime

import requests
import websocket

# ESPN endpoints
WS_HOST = "https://fastcast.semfs.engsvc.go.com/public/websockethost"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CHANNEL_PREFIX = "gp-basketball-nba-"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Stats
msg_count = 0
start_time = None
last_msg_time = None
message_types = {}


def find_game(team_query=None):
    """Find a live ESPN game. Returns (game_id, short_name) or None."""
    resp = requests.get(SCOREBOARD_URL, headers=HEADERS, timeout=10)
    events = resp.json().get("events", [])

    live_games = []
    for e in events:
        status_type = e.get("status", {}).get("type", {}).get("name", "")
        status_desc = e.get("status", {}).get("type", {}).get("description", "")
        if status_type == "STATUS_IN_PROGRESS":
            live_games.append({
                "id": e["id"],
                "name": e.get("shortName", e.get("name", "?")),
                "status": status_desc,
            })

    if not live_games:
        print("No live NBA games right now.")
        # Show all games
        for e in events:
            status = e.get("status", {}).get("type", {}).get("description", "")
            print(f"  {e.get('shortName', '?'):20s} {status}")
        return None

    if team_query:
        team_query = team_query.lower()
        for g in live_games:
            if team_query in g["name"].lower():
                return g["id"], g["name"]
        print(f"No live game found matching '{team_query}'. Live games:")
        for g in live_games:
            print(f"  {g['name']} ({g['status']})")
        return None

    # No query — show all live and pick first
    if len(live_games) == 1:
        g = live_games[0]
        return g["id"], g["name"]

    print("Multiple live games. Pick one or specify a team name:")
    for i, g in enumerate(live_games):
        print(f"  [{i+1}] {g['name']} ({g['status']})")
    g = live_games[0]
    print(f"\nDefaulting to: {g['name']}")
    return g["id"], g["name"]


def get_ws_url():
    """Get the WebSocket connection URL from ESPN's Fastcast host."""
    resp = requests.get(WS_HOST, headers=HEADERS, timeout=10)
    info = resp.json()
    token = info["token"]
    ip = info["ip"]
    port = str(info["securePort"])
    url = f"wss://{ip}:{port}/FastcastService/pubsub/profiles/12000?TrafficManager-Token={token}"
    return url


def decode_payload(message):
    """Decode an ESPN message payload (compressed or plain JSON)."""
    pl_raw = message["pl"]
    if isinstance(pl_raw, str):
        pl = json.loads(pl_raw)
    else:
        pl = pl_raw

    # Some messages have ~c=0 with plain JSON list in pl
    if isinstance(pl, dict):
        inner = pl.get("pl")
        if isinstance(inner, list):
            return inner
        if pl.get("~c") != "0" and isinstance(inner, str):
            raw = base64.b64decode(inner)
            decompressed = zlib.decompress(raw)
            return json.loads(decompressed)
    return None


def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def on_open(ws):
    global start_time
    start_time = time.time()
    print(f"[{ts()}] Connected! Sending handshake...")
    ws.send(json.dumps({"op": "C"}))


def on_message(ws, raw):
    global msg_count, last_msg_time, message_types

    now = time.time()
    gap = f" (+{now - last_msg_time:.1f}s)" if last_msg_time else ""
    last_msg_time = now
    msg_count += 1

    message = json.loads(raw)
    op = message.get("op", "?")

    # Track message types
    message_types[op] = message_types.get(op, 0) + 1

    if op == "C":
        # Handshake response — subscribe to game channel
        sid = message.get("sid", "")
        channel = CHANNEL_PREFIX + game_id
        print(f"[{ts()}] Handshake OK (sid={sid[:20]}...). Subscribing to channel: {channel}")
        ws.send(json.dumps({"op": "S", "sid": sid, "tc": channel}))

    elif op == "S":
        print(f"[{ts()}] Subscribed! Waiting for game events...\n")

    elif "pl" in message:
        tc = message.get("tc", "")
        pl_raw = message.get("pl", "0")

        if pl_raw == "0" or tc != CHANNEL_PREFIX + game_id:
            # Heartbeat or wrong channel
            return

        # Skip checkpoint URLs (initial state download links)
        if isinstance(pl_raw, str) and pl_raw.startswith("http"):
            print(f"[{ts()}] Checkpoint URL: {pl_raw[:120]}")
            return

        try:
            events = decode_payload(message)
            if not events:
                return

            for event in events:
                event_op = event.get("op", "?")
                path = event.get("path", "")
                value = event.get("value", "")

                if event_op == "add" and isinstance(value, dict) and "text" in value:
                    # Play-by-play event with score!
                    home_score = value.get("homeScore", "?")
                    away_score = value.get("awayScore", "?")
                    period = value.get("period", {}).get("number", "?")
                    clock = value.get("clock", {}).get("displayValue", "?")
                    text = value.get("text", "")
                    shooter = value.get("participants", [{}])[0].get("athlete", {}).get("displayName", "") if value.get("participants") else ""

                    print(f"[{ts()}]{gap} PLAY | Q{period} {clock:>5s} | "
                          f"{game_name.split(' @ ')[0]} {away_score} - {game_name.split(' @ ')[1]} {home_score} | "
                          f"{text}")

                    if text.lower() == "end of game":
                        print(f"\nGame over! Closing connection.")
                        ws.close()

                elif event_op == "replace":
                    # Score/stat updates (not play-by-play text)
                    if "score" in str(path).lower() or "score" in str(value).lower():
                        print(f"[{ts()}]{gap} SCORE UPDATE | path={path} | value={json.dumps(value)[:200]}")
                    elif isinstance(value, dict):
                        keys = list(value.keys())[:5]
                        print(f"[{ts()}]{gap} UPDATE | path={path} | keys={keys}")

                elif event_op == "add" and isinstance(value, dict):
                    keys = list(value.keys())[:8]
                    print(f"[{ts()}]{gap} ADD | path={path} | keys={keys}")

                else:
                    # Log unknown event types
                    val_preview = str(value)[:100] if value else ""
                    if val_preview and val_preview != "0":
                        print(f"[{ts()}]{gap} {event_op.upper()} | path={path} | {val_preview}")

        except Exception as e:
            print(f"[{ts()}] Decode error: {e}")
            # Print raw for debugging
            print(f"  Raw pl type: {type(pl_raw)}, first 200 chars: {str(pl_raw)[:200]}")

    elif op == "H":
        # Heartbeat
        pass

    else:
        print(f"[{ts()}]{gap} OP={op} | {json.dumps(message)[:200]}")


def on_error(ws, error):
    print(f"[{ts()}] ERROR: {error}")


def on_close(ws, code, reason):
    elapsed = time.time() - start_time if start_time else 0
    print(f"\n[{ts()}] Connection closed (code={code}, reason={reason})")
    print(f"  Total messages: {msg_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Avg msg rate: {msg_count/max(elapsed,1):.1f} msg/s")
    print(f"  Message types: {message_types}")


if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else None

    result = find_game(team)
    if not result:
        sys.exit(1)

    game_id, game_name = result
    print(f"\nGame: {game_name} (ESPN ID: {game_id})")

    ws_url = get_ws_url()
    print(f"WebSocket URL: {ws_url[:80]}...")
    print(f"Channel: {CHANNEL_PREFIX}{game_id}")
    print(f"Connecting...\n")

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        ws.close()
