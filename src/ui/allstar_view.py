"""All-Star Weekend Betting Helper.

Provides analytical rankings and value detection for:
- All-Star Game MVP
- 3-Point Contest
- Rising Stars Tournament
- All-Star Game Winner

Uses player stats from the database and lets users input betting
odds to calculate implied probability vs model probability.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.database.db import get_conn
from src.data.image_cache import get_player_photo_pixmap, get_team_logo_pixmap


# ════════════════════════════════════════════════════════════════════════
#  Odds conversion helpers
# ════════════════════════════════════════════════════════════════════════

def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> str:
    """Convert probability (0-1) to American odds string."""
    if prob <= 0 or prob >= 1:
        return "—"
    if prob >= 0.5:
        return f"{int(-prob / (1 - prob) * 100):+d}"
    else:
        return f"+{int((1 - prob) / prob * 100)}"


def edge_color(edge: float) -> QColor:
    """Return a color based on the edge (positive = green, negative = red)."""
    if edge > 10:
        return QColor("#10b981")  # strong value
    elif edge > 3:
        return QColor("#34d399")  # moderate value
    elif edge > 0:
        return QColor("#6ee7b7")  # slight value
    elif edge > -5:
        return QColor("#94a3b8")  # neutral
    else:
        return QColor("#f87171")  # negative value


# ════════════════════════════════════════════════════════════════════════
#  Data loading (runs in background thread)
# ════════════════════════════════════════════════════════════════════════

class _StatsWorker(QThread):
    """Load player season stats from the database."""
    finished = Signal(list)  # list of dicts

    def __init__(self, player_ids: List[int], recent_n: int = 0):
        super().__init__()
        self.player_ids = player_ids
        self.recent_n = recent_n

    def run(self) -> None:
        try:
            stats = _load_player_stats(self.player_ids, self.recent_n)
            self.finished.emit(stats)
        except Exception:
            self.finished.emit([])


def _load_player_stats(
    player_ids: List[int],
    recent_n: int = 0,
) -> List[Dict]:
    """Load aggregated season stats for a list of player IDs.

    If *recent_n* > 0, only use the most recent N games.
    """
    if not player_ids:
        return []
    results = []
    with get_conn() as conn:
        for pid in player_ids:
            # Player info
            p_row = conn.execute(
                "SELECT name, team_id, position FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if not p_row:
                continue
            name, team_id, position = str(p_row[0]), int(p_row[1]), str(p_row[2] or "")

            # Team abbreviation
            t_row = conn.execute(
                "SELECT abbreviation FROM teams WHERE team_id = ?",
                (team_id,),
            ).fetchone()
            team_abbr = str(t_row[0]) if t_row else "?"

            # Game logs
            order = "ORDER BY game_date DESC"
            limit = f"LIMIT {recent_n}" if recent_n > 0 else ""
            rows = conn.execute(
                f"""
                SELECT points, rebounds, assists, minutes, steals, blocks,
                       turnovers, fg_made, fg_attempted, fg3_made, fg3_attempted,
                       ft_made, ft_attempted, plus_minus
                FROM player_stats
                WHERE player_id = ?
                {order} {limit}
                """,
                (pid,),
            ).fetchall()

            if not rows:
                continue

            n = len(rows)
            pts = sum(r[0] or 0 for r in rows) / n
            reb = sum(r[1] or 0 for r in rows) / n
            ast = sum(r[2] or 0 for r in rows) / n
            mpg = sum(r[3] or 0 for r in rows) / n
            stl = sum(r[4] or 0 for r in rows) / n
            blk = sum(r[5] or 0 for r in rows) / n
            tov = sum(r[6] or 0 for r in rows) / n
            fgm = sum(r[7] or 0 for r in rows)
            fga = sum(r[8] or 0 for r in rows)
            fg3m = sum(r[9] or 0 for r in rows)
            fg3a = sum(r[10] or 0 for r in rows)
            ftm = sum(r[11] or 0 for r in rows)
            fta = sum(r[12] or 0 for r in rows)
            pm = sum(r[13] or 0 for r in rows) / n

            fg_pct = (fgm / fga * 100) if fga > 0 else 0
            fg3_pct = (fg3m / fg3a * 100) if fg3a > 0 else 0
            ft_pct = (ftm / fta * 100) if fta > 0 else 0
            fg3_per_game = fg3m / n if n > 0 else 0
            fg3a_per_game = fg3a / n if n > 0 else 0

            results.append({
                "player_id": pid,
                "name": name,
                "team_id": team_id,
                "team_abbr": team_abbr,
                "position": position,
                "games": n,
                "ppg": pts,
                "rpg": reb,
                "apg": ast,
                "mpg": mpg,
                "spg": stl,
                "bpg": blk,
                "topg": tov,
                "fg_pct": fg_pct,
                "fg3_pct": fg3_pct,
                "ft_pct": ft_pct,
                "fg3m_pg": fg3_per_game,
                "fg3a_pg": fg3a_per_game,
                "plus_minus": pm,
                # Composite scores (computed later per-event)
            })
    return results


def _load_all_players() -> List[Dict]:
    """Load basic info for all players in the database (for search/selection)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.player_id, p.name, p.team_id, t.abbreviation, p.position
            FROM players p
            LEFT JOIN teams t ON t.team_id = p.team_id
            ORDER BY p.name
            """,
        ).fetchall()
    return [
        {
            "player_id": int(r[0]),
            "name": str(r[1]),
            "team_id": int(r[2]),
            "team_abbr": str(r[3] or "?"),
            "position": str(r[4] or ""),
        }
        for r in rows
    ]


def _find_player_ids(names: List[str]) -> List[int]:
    """Look up player IDs by name (fuzzy last-name match)."""
    with get_conn() as conn:
        all_rows = conn.execute(
            "SELECT player_id, name FROM players"
        ).fetchall()
    # Build lookup: lowercase name -> player_id
    name_map = {str(r[1]).lower(): int(r[0]) for r in all_rows}
    # Also build last-name map for fuzzy matching
    last_map: Dict[str, int] = {}
    for r in all_rows:
        parts = str(r[1]).lower().split()
        if parts:
            last_map[parts[-1]] = int(r[0])

    ids = []
    for name in names:
        low = name.lower().strip()
        # Try exact match first
        if low in name_map:
            ids.append(name_map[low])
            continue
        # Try partial match (name contains search)
        found = False
        for full_name, pid in name_map.items():
            if low in full_name or full_name in low:
                ids.append(pid)
                found = True
                break
        if found:
            continue
        # Try last name only
        last = low.split()[-1] if low.split() else low
        if last in last_map:
            ids.append(last_map[last])
    return ids


# ════════════════════════════════════════════════════════════════════════
#  2026 All-Star Weekend — Prefill Data
#  (Participants and betting odds from FanDuel / PrizePicks)
#  Event: February 13-15, 2026 at Intuit Dome, Inglewood, CA
#  Format: USA vs. World (three teams, round-robin tournament)
# ════════════════════════════════════════════════════════════════════════

# ASG MVP candidates with FanDuel odds (as of Feb 13, 2026)
# Note: LeBron James (injury), Giannis Antetokounmpo (injury),
#       and Shai Gilgeous-Alexander (injury) are OUT.
#       De'Aaron Fox replaces Giannis on Team World.
ASG_MVP_PLAYERS = [
    # Team World
    "Victor Wembanyama", "Nikola Jokic", "Luka Doncic",
    "Jamal Murray", "Karl-Anthony Towns", "Pascal Siakam",
    "Deni Avdija", "De'Aaron Fox",
    # USA Stripes (veterans)
    "Stephen Curry", "Kevin Durant", "Jaylen Brown", "Jalen Brunson",
    "Donovan Mitchell", "Kawhi Leonard", "Norman Powell",
    # USA Stars (younger)
    "Anthony Edwards", "Tyrese Maxey", "Devin Booker",
    "Cade Cunningham", "Scottie Barnes", "Chet Holmgren",
    "Jalen Johnson", "Jalen Duren",
]
ASG_MVP_ODDS: Dict[str, int] = {
    "Victor Wembanyama": 390,
    "Tyrese Maxey": 900,
    "Jaylen Brown": 1000,
    "Jalen Brunson": 1000,
    "Devin Booker": 1100,
    "Cade Cunningham": 1100,
    "Jamal Murray": 1300,
    "Kevin Durant": 1300,
    "Donovan Mitchell": 1400,
    "Anthony Edwards": 1400,
    "Stephen Curry": 1600,
    "Luka Doncic": 1800,
    "Nikola Jokic": 2000,
    "Scottie Barnes": 2500,
    "Kawhi Leonard": 2500,
    "Karl-Anthony Towns": 3000,
    "Norman Powell": 3500,
    "Chet Holmgren": 4000,
    "Pascal Siakam": 4000,
    "De'Aaron Fox": 4000,
    "Jalen Johnson": 5000,
    "Deni Avdija": 5000,
    "Jalen Duren": 6000,
}

# 3-Point Contest participants with FanDuel odds (Feb 13, 2026)
# Saturday Feb 14 at Intuit Dome — 8 shooters
THREE_PT_PLAYERS = [
    "Kon Knueppel", "Damian Lillard", "Devin Booker", "Jamal Murray",
    "Tyrese Maxey", "Donovan Mitchell", "Norman Powell", "Bobby Portis",
]
THREE_PT_ODDS: Dict[str, int] = {
    "Kon Knueppel": 270,
    "Damian Lillard": 410,
    "Devin Booker": 550,
    "Jamal Murray": 650,
    "Tyrese Maxey": 650,
    "Donovan Mitchell": 750,
    "Norman Powell": 950,
    "Bobby Portis": 1600,
}

# Rising Stars rosters (Feb 13, 2026 at Intuit Dome)
RISING_STARS_PLAYERS = {
    "Team Melo": [
        "Reed Sheppard", "Stephon Castle", "Dylan Harper",
        "Jeremiah Fears", "Donovan Clingan", "Collin Murray-Boyles",
        "Ace Bailey",
    ],
    "Team T-Mac": [
        "Kon Knueppel", "Kel'el Ware", "Tre Johnson",
        "Jaylon Tyson", "Cam Spencer", "Bub Carrington",
        "Zaccharie Risacher",
    ],
    "Team Vince": [
        "VJ Edgecombe", "Derik Queen", "Kyshawn George",
        "Matas Buzelis", "Egor Demin", "Jaylen Wells", "Carter Bryant",
    ],
    "Team Austin (G League)": [
        "Sean East II", "Ron Harper Jr.", "Alijah Martin",
        "Tristen Newton", "Yang Hansen", "Jahmir Young",
    ],
}
RISING_STARS_TEAM_ODDS: Dict[str, int] = {
    "Team Melo": 200, "Team T-Mac": 275,
    "Team Vince": 350, "Team Austin (G League)": 500,
}

# All-Star Game team rosters (2026 USA vs. World format)
ASG_TEAMS = {
    "Team World": [
        "Victor Wembanyama", "Nikola Jokic", "Luka Doncic",
        "Jamal Murray", "Karl-Anthony Towns", "Pascal Siakam",
        "Deni Avdija", "De'Aaron Fox",
    ],
    "USA Stripes (Veterans)": [
        "Stephen Curry", "Kevin Durant", "Jaylen Brown", "Jalen Brunson",
        "Kawhi Leonard", "Donovan Mitchell", "Norman Powell",
    ],
    "USA Stars (Young)": [
        "Anthony Edwards", "Tyrese Maxey", "Devin Booker",
        "Cade Cunningham", "Scottie Barnes", "Chet Holmgren",
        "Jalen Johnson", "Jalen Duren",
    ],
}
ASG_TEAM_ODDS: Dict[str, int] = {
    "Team World": 155,
    "USA Stripes (Veterans)": 160,
    "USA Stars (Young)": 200,
}


# ════════════════════════════════════════════════════════════════════════
#  Scoring models
# ════════════════════════════════════════════════════════════════════════

def compute_mvp_score(stat: Dict) -> float:
    """Score a player for ASG MVP probability.

    Historical ASG MVPs share these traits:
    - High scoring (almost always 25+ pts in the game)
    - Highlight plays (blocks, 3-pointers, lobs/dunks)
    - On the winning team (can't model this, but +/- is a proxy)
    - "Wow factor" — bigs who shoot 3s, guards who block, etc.

    The model balances scoring volume, defensive highlights (blocks,
    steals), two-way versatility, and efficiency.  Blocks are weighted
    heavily because they're the most dramatic ASG highlight.
    """
    pts = stat.get("ppg", 0)
    fg3 = stat.get("fg3m_pg", 0)
    ast = stat.get("apg", 0)
    reb = stat.get("rpg", 0)
    blk = stat.get("bpg", 0)
    stl = stat.get("spg", 0)
    eff = stat.get("fg_pct", 0) / 100
    ft = stat.get("ft_pct", 0) / 100
    mpg = stat.get("mpg", 0)
    pm = stat.get("plus_minus", 0)
    pos = stat.get("position", "")

    # ── Scoring volume (most important — MVP almost always top scorer)
    score = pts * 2.5

    # ── Highlight plays ──
    # Blocks are THE biggest ASG highlight — crowd goes wild
    score += blk * 8.0
    # 3-point makes (especially from unexpected positions)
    score += fg3 * 3.5
    # Steals → fast-break highlights
    score += stl * 4.0

    # ── Playmaking (flashy assists = highlights)
    score += ast * 1.8

    # ── Rebounds (dominance on the boards, especially for bigs)
    score += reb * 1.0

    # ── Efficiency bonus
    score += eff * 12
    score += ft * 4

    # ── Winner proxy (MVP comes from winning team)
    if pm > 0:
        score += min(pm, 8) * 1.5

    # ── Minutes proxy for star status / likely ASG minutes
    score += min(mpg, 36) * 0.4

    # ── "Unicorn" bonus: bigs who shoot 3s get extra highlight value
    # (e.g., Wembanyama, KAT — a 7-footer draining 3s is electric)
    is_big = any(p in pos.upper() for p in ("C", "CENTER", "PF", "FORWARD"))
    if is_big and fg3 >= 1.0:
        score += fg3 * 4.0  # extra bonus on top of base 3PT credit

    # ── Two-way versatility bonus (pts + blk + ast + stl all above thresholds)
    versatile_count = sum([
        pts >= 20, blk >= 1.5, ast >= 3.0, stl >= 1.0, reb >= 8.0,
    ])
    if versatile_count >= 4:
        score += 15.0  # elite two-way players dominate ASG format
    elif versatile_count >= 3:
        score += 8.0

    return max(0, score)


def compute_three_pt_score(stat: Dict) -> float:
    """Score a player for 3-Point Contest probability.

    Based on: 3PT%, 3PT volume, consistency, FT% (shooting touch).
    """
    fg3_pct = stat.get("fg3_pct", 0)
    fg3m = stat.get("fg3m_pg", 0)
    fg3a = stat.get("fg3a_pg", 0)
    ft_pct = stat.get("ft_pct", 0)

    # Accuracy is king in the contest
    score = fg3_pct * 2.0
    # Volume shows confidence and rhythm
    score += fg3m * 10.0
    # Attempts show shooting role
    score += fg3a * 2.0
    # FT% correlates with shooting form
    score += ft_pct * 0.5

    return max(0, score)


def compute_rising_star_score(stat: Dict) -> float:
    """Score a young player for Rising Stars impact.

    Balanced scoring emphasising overall production + efficiency.
    """
    pts = stat.get("ppg", 0)
    reb = stat.get("rpg", 0)
    ast = stat.get("apg", 0)
    stl = stat.get("spg", 0)
    blk = stat.get("bpg", 0)
    eff = stat.get("fg_pct", 0) / 100
    mpg = stat.get("mpg", 0)

    score = pts * 2.0
    score += reb * 1.2
    score += ast * 1.5
    score += stl * 3.0
    score += blk * 3.0
    score += eff * 20
    score += min(mpg, 36) * 0.3

    return max(0, score)


def scores_to_probabilities(scores: List[float]) -> List[float]:
    """Convert raw scores to probabilities using softmax with temperature."""
    if not scores or all(s == 0 for s in scores):
        n = len(scores) if scores else 1
        return [1.0 / n] * n

    temperature = 15.0  # lower = more decisive
    max_s = max(scores)
    exps = [math.exp((s - max_s) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ════════════════════════════════════════════════════════════════════════
#  Reusable player picker widget
# ════════════════════════════════════════════════════════════════════════

class PlayerPicker(QWidget):
    """Search + select widget for picking multiple NBA players."""
    players_changed = Signal()

    def __init__(self, label: str = "Players", max_players: int = 30) -> None:
        super().__init__()
        self._all_players: List[Dict] = []
        self._selected_ids: List[int] = []
        self._max = max_players

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Search row
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText(f"Search player to add ({label})...")
        self._search.textChanged.connect(self._filter_dropdown)
        self._dropdown = QComboBox()
        self._dropdown.setMinimumWidth(220)
        self._dropdown.setMaxVisibleItems(12)
        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(60)
        add_btn.clicked.connect(self._add_selected)
        search_row.addWidget(self._search, 2)
        search_row.addWidget(self._dropdown, 2)
        search_row.addWidget(add_btn)
        layout.addLayout(search_row)

        # Selected players display
        self._selected_label = QLabel("Selected: none")
        self._selected_label.setWordWrap(True)
        self._selected_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self._selected_label)

        # Clear button
        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedWidth(80)
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

    def load_players(self) -> None:
        """Load all players into the dropdown (call once)."""
        self._all_players = _load_all_players()
        self._populate_dropdown(self._all_players)

    def _populate_dropdown(self, players: List[Dict]) -> None:
        self._dropdown.blockSignals(True)
        self._dropdown.clear()
        for p in players[:200]:  # Limit for performance
            label = f"{p['name']} ({p['team_abbr']}) - {p['position']}"
            self._dropdown.addItem(label, p["player_id"])
        self._dropdown.blockSignals(False)

    def _filter_dropdown(self, text: str) -> None:
        text = text.lower().strip()
        if not text:
            self._populate_dropdown(self._all_players)
            return
        filtered = [p for p in self._all_players if text in p["name"].lower()
                     or text in p["team_abbr"].lower()]
        self._populate_dropdown(filtered)

    def _add_selected(self) -> None:
        pid = self._dropdown.currentData()
        if pid and pid not in self._selected_ids and len(self._selected_ids) < self._max:
            self._selected_ids.append(pid)
            self._update_label()
            self.players_changed.emit()

    def clear(self) -> None:
        self._selected_ids.clear()
        self._update_label()
        self.players_changed.emit()

    def selected_ids(self) -> List[int]:
        return list(self._selected_ids)

    def set_player_ids(self, ids: List[int]) -> None:
        """Programmatically set selected player IDs."""
        self._selected_ids = list(ids)
        self._update_label()
        self.players_changed.emit()

    def _update_label(self) -> None:
        if not self._selected_ids:
            self._selected_label.setText("Selected: none")
            return
        # Look up names
        names = []
        for pid in self._selected_ids:
            match = next((p for p in self._all_players if p["player_id"] == pid), None)
            if match:
                names.append(f"{match['name']} ({match['team_abbr']})")
            else:
                names.append(f"ID {pid}")
        self._selected_label.setText(f"Selected ({len(names)}): " + ", ".join(names))


# ════════════════════════════════════════════════════════════════════════
#  Results table with odds input
# ════════════════════════════════════════════════════════════════════════

class BettingTable(QTableWidget):
    """Table showing player rankings with editable odds column."""

    COLUMNS = [
        "Rank", "Player", "Team", "Key Stat", "Model %",
        "Fair Odds", "Your Odds", "Implied %", "Edge %", "Rating",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.setMinimumHeight(300)

    def populate(
        self,
        stats: List[Dict],
        score_fn,
        key_stat_fn,
        odds_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """Fill the table with ranked players.

        *score_fn(stat) -> float*: scoring function
        *key_stat_fn(stat) -> str*: format the main stat for display
        *odds_map*: {player_id: american_odds} from user input
        """
        if not stats:
            self.setRowCount(0)
            return

        # Compute scores
        for s in stats:
            s["_score"] = score_fn(s)

        ranked = sorted(stats, key=lambda s: s["_score"], reverse=True)
        scores = [s["_score"] for s in ranked]
        probs = scores_to_probabilities(scores)

        self.setRowCount(len(ranked))
        odds_map = odds_map or {}

        for i, (stat, prob) in enumerate(zip(ranked, probs)):
            pid = stat["player_id"]
            user_odds = odds_map.get(pid, 0)
            implied = american_to_implied(user_odds) if user_odds != 0 else 0.0
            edge = (prob - implied) * 100 if implied > 0 else 0.0

            # Rating
            if edge > 10:
                rating = "STRONG VALUE"
            elif edge > 3:
                rating = "VALUE"
            elif edge > 0:
                rating = "slight +"
            elif implied == 0:
                rating = "—"
            elif edge > -5:
                rating = "fair"
            else:
                rating = "AVOID"

            items = [
                str(i + 1),
                stat["name"],
                stat["team_abbr"],
                key_stat_fn(stat),
                f"{prob * 100:.1f}%",
                implied_to_american(prob),
                f"{user_odds:+d}" if user_odds != 0 else "—",
                f"{implied * 100:.1f}%" if implied > 0 else "—",
                f"{edge:+.1f}%" if implied > 0 else "—",
                rating,
            ]

            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color coding
                if col == 4:  # Model %
                    if prob > 0.15:
                        item.setForeground(QColor("#10b981"))
                    elif prob > 0.08:
                        item.setForeground(QColor("#fbbf24"))
                if col == 8 and implied > 0:  # Edge
                    item.setForeground(edge_color(edge))
                if col == 9:  # Rating
                    if "STRONG" in rating:
                        item.setForeground(QColor("#10b981"))
                        f = item.font()
                        f.setBold(True)
                        item.setFont(f)
                    elif "VALUE" in rating:
                        item.setForeground(QColor("#34d399"))
                    elif "AVOID" in rating:
                        item.setForeground(QColor("#f87171"))

                self.setItem(i, col, item)


# ════════════════════════════════════════════════════════════════════════
#  Individual event panels
# ════════════════════════════════════════════════════════════════════════

class _OddsInputRow(QWidget):
    """Row with player name + odds spin box for entering betting lines."""
    odds_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._entries: Dict[int, QSpinBox] = {}
        self._layout = QGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.setLayout(self._layout)

    def set_players(self, stats: List[Dict]) -> None:
        """Rebuild the odds input grid for the given players."""
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._entries.clear()

        cols = 4  # players per row
        for i, s in enumerate(stats):
            row, col = divmod(i, cols)
            container = QWidget()
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)

            lbl = QLabel(f"{s['name'][:18]}:")
            lbl.setFixedWidth(130)
            lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            spin = QSpinBox()
            spin.setRange(-50000, 50000)
            spin.setValue(0)
            spin.setSpecialValueText("—")
            spin.setPrefix("")
            spin.setToolTip(f"Enter American odds for {s['name']}")
            spin.setFixedWidth(90)
            spin.valueChanged.connect(self.odds_changed.emit)

            h.addWidget(lbl)
            h.addWidget(spin)
            container.setLayout(h)

            self._layout.addWidget(container, row, col)
            self._entries[s["player_id"]] = spin

    def get_odds_map(self) -> Dict[int, int]:
        return {pid: spin.value() for pid, spin in self._entries.items()
                if spin.value() != 0}


class MVPPanel(QWidget):
    """All-Star Game MVP analysis panel."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Header
        header = QLabel("All-Star Game MVP Predictor")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)

        desc = QLabel(
            "Model factors: scoring volume, blocks (biggest ASG highlight), 3PT makes, "
            "assists, steals, rebounds, efficiency, two-way versatility, and unicorn "
            "bonus (bigs who shoot 3s like Wemby/KAT get extra credit)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Player picker + prefill button
        picker_row = QHBoxLayout()
        self._picker = PlayerPicker("MVP Candidates", max_players=24)
        picker_row.addWidget(self._picker, 1)
        prefill_btn = QPushButton("Load 2026 All-Stars + Odds")
        prefill_btn.setToolTip("Pre-fill with 2026 ASG participants and FanDuel odds")
        prefill_btn.setFixedWidth(200)
        prefill_btn.clicked.connect(self._prefill)
        picker_row.addWidget(prefill_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(picker_row)

        # Recent games filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Recent games:"))
        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(0, 82)
        self._recent_spin.setValue(15)
        self._recent_spin.setSpecialValueText("Full season")
        self._recent_spin.setToolTip("0 = full season, or limit to last N games")
        filter_row.addWidget(self._recent_spin)
        filter_row.addStretch()

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(100)
        analyze_btn.clicked.connect(self._analyze)
        filter_row.addWidget(analyze_btn)
        layout.addLayout(filter_row)

        # Odds input
        odds_group = QGroupBox("Enter Betting Odds (American format, e.g. +500, -150)")
        self._odds_input = _OddsInputRow()
        self._odds_input.odds_changed.connect(self._refresh_table)
        odds_layout = QVBoxLayout()
        odds_scroll = QScrollArea()
        odds_scroll.setWidgetResizable(True)
        odds_scroll.setWidget(self._odds_input)
        odds_scroll.setMaximumHeight(140)
        odds_layout.addWidget(odds_scroll)
        odds_group.setLayout(odds_layout)
        layout.addWidget(odds_group)

        # Results table
        self._table = BettingTable()
        layout.addWidget(self._table, 1)

        # Insights
        self._insights = QLabel("")
        self._insights.setWordWrap(True)
        self._insights.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 4px; "
            "background: #172333; border-radius: 4px;"
        )
        layout.addWidget(self._insights)

        self.setLayout(layout)
        self._stats: List[Dict] = []
        self._worker: Optional[_StatsWorker] = None

    def init_data(self) -> None:
        self._picker.load_players()

    def _prefill(self) -> None:
        """Load 2025 All-Star participants and betting odds."""
        ids = _find_player_ids(ASG_MVP_PLAYERS)
        if ids:
            self._picker.set_player_ids(ids)
            self._prefill_odds = True  # flag to auto-fill odds after stats load
            self._analyze()
        else:
            self._insights.setText("Could not find All-Star players in database. Run a data sync first.")

    def _analyze(self) -> None:
        ids = self._picker.selected_ids()
        if not ids:
            self._insights.setText("Add players using the search above, or click 'Load 2025 All-Stars + Odds'.")
            return
        recent = self._recent_spin.value()
        self._worker = _StatsWorker(ids, recent)
        self._worker.finished.connect(self._on_stats_loaded)
        self._worker.start()
        self._insights.setText("Loading stats...")

    def _on_stats_loaded(self, stats: List[Dict]) -> None:
        self._stats = stats
        self._odds_input.set_players(stats)
        # Auto-fill odds from prefill data
        if getattr(self, "_prefill_odds", False):
            self._prefill_odds = False
            self._auto_fill_odds(stats, ASG_MVP_ODDS)
        self._refresh_table()

    def _auto_fill_odds(self, stats: List[Dict], odds_dict: Dict[str, int]) -> None:
        """Set odds spin boxes from the prefill odds dictionary."""
        for s in stats:
            name = s["name"]
            # Try exact match, then partial
            odds_val = odds_dict.get(name, 0)
            if not odds_val:
                for oname, oval in odds_dict.items():
                    if oname.lower() in name.lower() or name.lower() in oname.lower():
                        odds_val = oval
                        break
                    # Last name match
                    if oname.split()[-1].lower() == name.split()[-1].lower():
                        odds_val = oval
                        break
            if odds_val and s["player_id"] in self._odds_input._entries:
                self._odds_input._entries[s["player_id"]].setValue(odds_val)

    def _refresh_table(self) -> None:
        odds = self._odds_input.get_odds_map()
        self._table.populate(
            self._stats,
            compute_mvp_score,
            lambda s: (
                f"{s['ppg']:.1f}p / {s['bpg']:.1f}b / {s['fg3m_pg']:.1f}3 / "
                f"{s['apg']:.1f}a / {s['rpg']:.1f}r"
            ),
            odds,
        )
        # Generate insights
        if self._stats:
            top = sorted(self._stats, key=compute_mvp_score, reverse=True)
            best = top[0]
            self._insights.setText(
                f"Top pick: {best['name']} ({best['team_abbr']}) — "
                f"{best['ppg']:.1f} PPG, {best['bpg']:.1f} BPG, "
                f"{best['fg3m_pg']:.1f} 3PM/G, {best['apg']:.1f} APG, "
                f"{best['rpg']:.1f} RPG. "
                f"ASG MVP favors highlight scorers + two-way dominance. "
                f"Blocks and 3s from bigs are the biggest crowd-pleasers."
            )


class ThreePointPanel(QWidget):
    """3-Point Contest analysis panel."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        header = QLabel("3-Point Contest Predictor")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)

        desc = QLabel(
            "Ranks shooters by 3PT accuracy, volume, and shooting form (FT%). "
            "Pure shooters with high 3PT% on good volume have the edge."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        picker_row = QHBoxLayout()
        self._picker = PlayerPicker("3PT Contestants", max_players=8)
        picker_row.addWidget(self._picker, 1)
        prefill_btn = QPushButton("Load 2026 Contest + Odds")
        prefill_btn.setFixedWidth(200)
        prefill_btn.clicked.connect(self._prefill)
        picker_row.addWidget(prefill_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(picker_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Recent games:"))
        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(0, 82)
        self._recent_spin.setValue(20)
        self._recent_spin.setSpecialValueText("Full season")
        filter_row.addWidget(self._recent_spin)
        filter_row.addStretch()

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(100)
        analyze_btn.clicked.connect(self._analyze)
        filter_row.addWidget(analyze_btn)
        layout.addLayout(filter_row)

        odds_group = QGroupBox("Enter Betting Odds")
        self._odds_input = _OddsInputRow()
        self._odds_input.odds_changed.connect(self._refresh_table)
        odds_layout = QVBoxLayout()
        odds_scroll = QScrollArea()
        odds_scroll.setWidgetResizable(True)
        odds_scroll.setWidget(self._odds_input)
        odds_scroll.setMaximumHeight(120)
        odds_layout.addWidget(odds_scroll)
        odds_group.setLayout(odds_layout)
        layout.addWidget(odds_group)

        self._table = BettingTable()
        layout.addWidget(self._table, 1)

        self._insights = QLabel("")
        self._insights.setWordWrap(True)
        self._insights.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 4px; "
            "background: #172333; border-radius: 4px;"
        )
        layout.addWidget(self._insights)

        self.setLayout(layout)
        self._stats: List[Dict] = []
        self._worker: Optional[_StatsWorker] = None

    def init_data(self) -> None:
        self._picker.load_players()

    def _prefill(self) -> None:
        ids = _find_player_ids(THREE_PT_PLAYERS)
        if ids:
            self._picker.set_player_ids(ids)
            self._prefill_odds = True
            self._analyze()
        else:
            self._insights.setText("Could not find 3PT contest players in database.")

    def _analyze(self) -> None:
        ids = self._picker.selected_ids()
        if not ids:
            self._insights.setText("Add 3-point contest participants or click 'Load 2025 Contest + Odds'.")
            return
        recent = self._recent_spin.value()
        self._worker = _StatsWorker(ids, recent)
        self._worker.finished.connect(self._on_stats_loaded)
        self._worker.start()
        self._insights.setText("Loading stats...")

    def _on_stats_loaded(self, stats: List[Dict]) -> None:
        self._stats = stats
        self._odds_input.set_players(stats)
        if getattr(self, "_prefill_odds", False):
            self._prefill_odds = False
            self._auto_fill_odds(stats, THREE_PT_ODDS)
        self._refresh_table()

    def _auto_fill_odds(self, stats: List[Dict], odds_dict: Dict[str, int]) -> None:
        for s in stats:
            name = s["name"]
            odds_val = odds_dict.get(name, 0)
            if not odds_val:
                for oname, oval in odds_dict.items():
                    if oname.split()[-1].lower() == name.split()[-1].lower():
                        odds_val = oval
                        break
            if odds_val and s["player_id"] in self._odds_input._entries:
                self._odds_input._entries[s["player_id"]].setValue(odds_val)

    def _refresh_table(self) -> None:
        odds = self._odds_input.get_odds_map()
        self._table.populate(
            self._stats,
            compute_three_pt_score,
            lambda s: f"{s['fg3_pct']:.1f}% 3PT / {s['fg3m_pg']:.1f} 3PM / {s['fg3a_pg']:.1f} 3PA",
            odds,
        )
        if self._stats:
            top = sorted(self._stats, key=compute_three_pt_score, reverse=True)
            best = top[0]
            self._insights.setText(
                f"Top pick: {best['name']} ({best['team_abbr']}) — "
                f"{best['fg3_pct']:.1f}% from three on {best['fg3a_pg']:.1f} attempts/game. "
                f"High accuracy + volume shooters dominate the contest."
            )


class RisingStarsPanel(QWidget):
    """Rising Stars Tournament analysis panel — team-level + individual."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        header = QLabel("Rising Stars Tournament Predictor")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)

        desc = QLabel(
            "Four-team tournament (first to 40 in semis, first to 25 in final). "
            "Model aggregates each team's combined stats to rank tournament winner "
            "probability. Individual player rankings are shown below for MVP/top scorer bets."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Controls row
        ctrl_row = QHBoxLayout()
        prefill_btn = QPushButton("Load 2026 Teams + Odds")
        prefill_btn.setFixedWidth(200)
        prefill_btn.clicked.connect(self._prefill)
        ctrl_row.addWidget(prefill_btn)
        ctrl_row.addWidget(QLabel("Recent games:"))
        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(0, 82)
        self._recent_spin.setValue(15)
        self._recent_spin.setSpecialValueText("Full season")
        ctrl_row.addWidget(self._recent_spin)
        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(100)
        analyze_btn.clicked.connect(self._prefill)
        ctrl_row.addWidget(analyze_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── TEAM COMPARISON TABLE ──
        team_group = QGroupBox("Team Rankings (Tournament Winner)")
        team_layout = QVBoxLayout()

        self._team_table = QTableWidget()
        team_cols = [
            "Rank", "Team", "Avg PPG", "Avg RPG", "Avg APG",
            "Avg FG%", "Avg 3PT%", "Team Score", "Model %",
            "Fair Odds", "Your Odds", "Implied %", "Edge %", "Rating",
        ]
        self._team_table.setColumnCount(len(team_cols))
        self._team_table.setHorizontalHeaderLabels(team_cols)
        self._team_table.setAlternatingRowColors(True)
        self._team_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._team_table.verticalHeader().setVisible(False)
        self._team_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        th = self._team_table.horizontalHeader()
        th.setStretchLastSection(True)
        th.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        th.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._team_table.setMaximumHeight(180)
        team_layout.addWidget(self._team_table)

        # Team odds input row
        odds_row = QHBoxLayout()
        self._team_odds_spins: Dict[str, QSpinBox] = {}
        for team_name in RISING_STARS_PLAYERS:
            lbl = QLabel(f"{team_name}:")
            lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            spin = QSpinBox()
            spin.setRange(-50000, 50000)
            spin.setValue(0)
            spin.setSpecialValueText("—")
            spin.setFixedWidth(90)
            spin.setToolTip(f"American odds for {team_name} to win tournament")
            spin.valueChanged.connect(self._refresh_teams)
            odds_row.addWidget(lbl)
            odds_row.addWidget(spin)
            self._team_odds_spins[team_name] = spin
        odds_row.addStretch()
        team_layout.addLayout(odds_row)

        team_group.setLayout(team_layout)
        layout.addWidget(team_group)

        # ── INDIVIDUAL PLAYER TABLE ──
        player_group = QGroupBox("Individual Player Rankings (MVP / Top Scorer)")
        player_layout = QVBoxLayout()
        self._table = BettingTable()
        player_layout.addWidget(self._table)
        player_group.setLayout(player_layout)
        layout.addWidget(player_group, 1)

        # Insights
        self._insights = QLabel("")
        self._insights.setWordWrap(True)
        self._insights.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 4px; "
            "background: #172333; border-radius: 4px;"
        )
        layout.addWidget(self._insights)

        self.setLayout(layout)
        self._team_stats: Dict[str, List[Dict]] = {}  # team_name -> [player stats]
        self._all_stats: List[Dict] = []
        self._workers: List[_StatsWorker] = []
        self._pending = 0

    def init_data(self) -> None:
        pass  # No picker to load — prefill handles it

    def _prefill(self) -> None:
        """Load all Rising Stars teams and analyze."""
        self._team_stats = {}
        self._all_stats = []
        self._pending = len(RISING_STARS_PLAYERS)
        self._workers = []

        recent = self._recent_spin.value()
        for team_name, players in RISING_STARS_PLAYERS.items():
            ids = _find_player_ids(players)
            if ids:
                w = _StatsWorker(ids, recent)
                w.finished.connect(lambda stats, tn=team_name: self._on_team_loaded(tn, stats))
                w.start()
                self._workers.append(w)
            else:
                self._pending -= 1
                self._team_stats[team_name] = []

        # Auto-fill team odds
        for team_name, odds_val in RISING_STARS_TEAM_ODDS.items():
            if team_name in self._team_odds_spins:
                self._team_odds_spins[team_name].setValue(odds_val)

        self._insights.setText("Loading team stats...")

    def _on_team_loaded(self, team_name: str, stats: List[Dict]) -> None:
        self._team_stats[team_name] = stats
        self._all_stats.extend(stats)
        self._pending -= 1
        if self._pending <= 0:
            self._refresh_teams()
            self._refresh_players()

    def _refresh_teams(self) -> None:
        """Update the team comparison table."""
        if not self._team_stats:
            return

        def _team_score(stats: List[Dict]) -> float:
            if not stats:
                return 0
            ppg = sum(s.get("ppg", 0) for s in stats) / len(stats)
            rpg = sum(s.get("rpg", 0) for s in stats) / len(stats)
            apg = sum(s.get("apg", 0) for s in stats) / len(stats)
            fg = sum(s.get("fg_pct", 0) for s in stats) / len(stats)
            fg3 = sum(s.get("fg3_pct", 0) for s in stats) / len(stats)
            stk = sum(s.get("spg", 0) + s.get("bpg", 0) for s in stats) / len(stats)
            return ppg * 2.0 + rpg * 1.0 + apg * 1.5 + fg * 0.3 + fg3 * 0.3 + stk * 3.0

        # Calculate scores and probabilities
        team_data = []
        for team_name, stats in self._team_stats.items():
            score = _team_score(stats)
            n = len(stats) if stats else 1
            ppg = sum(s.get("ppg", 0) for s in stats) / n if stats else 0
            rpg = sum(s.get("rpg", 0) for s in stats) / n if stats else 0
            apg = sum(s.get("apg", 0) for s in stats) / n if stats else 0
            fg = sum(s.get("fg_pct", 0) for s in stats) / n if stats else 0
            fg3 = sum(s.get("fg3_pct", 0) for s in stats) / n if stats else 0
            team_data.append({
                "name": team_name, "score": score,
                "ppg": ppg, "rpg": rpg, "apg": apg, "fg": fg, "fg3": fg3,
            })

        # Sort by score
        team_data.sort(key=lambda t: t["score"], reverse=True)
        scores = [t["score"] for t in team_data]
        probs = scores_to_probabilities(scores)

        self._team_table.setRowCount(len(team_data))

        for i, (td, prob) in enumerate(zip(team_data, probs)):
            tn = td["name"]
            user_odds = self._team_odds_spins.get(tn, None)
            user_odds_val = user_odds.value() if user_odds else 0
            implied = american_to_implied(user_odds_val) if user_odds_val != 0 else 0.0
            edge_val = (prob - implied) * 100 if implied > 0 else 0.0

            if edge_val > 10:
                rating = "STRONG VALUE"
            elif edge_val > 3:
                rating = "VALUE"
            elif edge_val > 0:
                rating = "slight +"
            elif implied == 0:
                rating = "—"
            elif edge_val > -5:
                rating = "fair"
            else:
                rating = "AVOID"

            items = [
                str(i + 1), tn,
                f"{td['ppg']:.1f}", f"{td['rpg']:.1f}", f"{td['apg']:.1f}",
                f"{td['fg']:.1f}%", f"{td['fg3']:.1f}%",
                f"{td['score']:.0f}", f"{prob * 100:.1f}%",
                implied_to_american(prob),
                f"{user_odds_val:+d}" if user_odds_val != 0 else "—",
                f"{implied * 100:.1f}%" if implied > 0 else "—",
                f"{edge_val:+.1f}%" if implied > 0 else "—",
                rating,
            ]

            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 8 and prob > 0.30:
                    item.setForeground(QColor("#10b981"))
                if col == 12 and implied > 0:
                    item.setForeground(edge_color(edge_val))
                if col == 13:
                    if "STRONG" in rating:
                        item.setForeground(QColor("#10b981"))
                        f = item.font()
                        f.setBold(True)
                        item.setFont(f)
                    elif "VALUE" in rating:
                        item.setForeground(QColor("#34d399"))
                    elif "AVOID" in rating:
                        item.setForeground(QColor("#f87171"))
                self._team_table.setItem(i, col, item)

        # Insights
        if team_data and probs:
            best = team_data[0]
            self._insights.setText(
                f"Strongest team: {best['name']} — "
                f"avg {best['ppg']:.1f} PPG, {best['rpg']:.1f} RPG, {best['apg']:.1f} APG. "
                f"Model win probability: {probs[0] * 100:.1f}%. "
                f"Short games (first to 40/25) favor teams with scorers who can get hot quickly."
            )

    def _refresh_players(self) -> None:
        """Update the individual player table."""
        self._table.populate(
            self._all_stats,
            compute_rising_star_score,
            lambda s: (
                f"{s['ppg']:.1f} PPG / {s['rpg']:.1f} RPG / "
                f"{s['apg']:.1f} APG / {s['spg'] + s['bpg']:.1f} STK"
            ),
        )


class AllStarGamePanel(QWidget):
    """All-Star Game Winner analysis panel."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        header = QLabel("All-Star Game Winner Predictor")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)

        desc = QLabel(
            "Compare rosters for each All-Star team. Model aggregates scoring, "
            "shooting, playmaking, and defensive metrics to project team strength. "
            "Use prefill buttons to load 2025 ASG team matchups."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Quick-load buttons for 2026 ASG matchups (USA vs World, 3 teams)
        prefill_row = QHBoxLayout()
        prefill_lbl = QLabel("2026 Matchups:")
        prefill_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        prefill_row.addWidget(prefill_lbl)
        load_g1 = QPushButton("Game 1: USA Stars vs World")
        load_g1.clicked.connect(self._prefill_game1)
        prefill_row.addWidget(load_g1)
        load_g2 = QPushButton("USA Stripes vs World")
        load_g2.clicked.connect(self._prefill_stripes_vs_world)
        prefill_row.addWidget(load_g2)
        load_g3 = QPushButton("USA Stars vs Stripes")
        load_g3.clicked.connect(self._prefill_stars_vs_stripes)
        prefill_row.addWidget(load_g3)
        prefill_row.addStretch()
        layout.addLayout(prefill_row)

        # Two team pickers side by side
        teams_row = QHBoxLayout()

        team1_box = QGroupBox("Team 1")
        t1_layout = QVBoxLayout()
        self._team1_picker = PlayerPicker("Team 1", max_players=12)
        t1_layout.addWidget(self._team1_picker)
        team1_box.setLayout(t1_layout)

        team2_box = QGroupBox("Team 2")
        t2_layout = QVBoxLayout()
        self._team2_picker = PlayerPicker("Team 2", max_players=12)
        t2_layout.addWidget(self._team2_picker)
        team2_box.setLayout(t2_layout)

        teams_row.addWidget(team1_box)
        teams_row.addWidget(team2_box)
        layout.addLayout(teams_row)

        # Controls
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Recent games:"))
        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(0, 82)
        self._recent_spin.setValue(15)
        self._recent_spin.setSpecialValueText("Full season")
        ctrl_row.addWidget(self._recent_spin)
        ctrl_row.addStretch()

        # Odds input for game winner
        ctrl_row.addWidget(QLabel("Team 1 odds:"))
        self._team1_odds = QSpinBox()
        self._team1_odds.setRange(-50000, 50000)
        self._team1_odds.setValue(0)
        self._team1_odds.setSpecialValueText("—")
        self._team1_odds.setFixedWidth(90)
        ctrl_row.addWidget(self._team1_odds)

        ctrl_row.addWidget(QLabel("Team 2 odds:"))
        self._team2_odds = QSpinBox()
        self._team2_odds.setRange(-50000, 50000)
        self._team2_odds.setValue(0)
        self._team2_odds.setSpecialValueText("—")
        self._team2_odds.setFixedWidth(90)
        ctrl_row.addWidget(self._team2_odds)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(100)
        analyze_btn.clicked.connect(self._analyze)
        ctrl_row.addWidget(analyze_btn)
        layout.addLayout(ctrl_row)

        # Comparison display
        self._comparison = QFrame()
        self._comparison.setStyleSheet(
            "background: #172333; border-radius: 8px; padding: 12px;"
        )
        comp_layout = QGridLayout()

        self._t1_header = QLabel("Team 1")
        self._t1_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._t1_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #60a5fa;")
        self._vs_label = QLabel("VS")
        self._vs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._vs_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f59e0b;")
        self._t2_header = QLabel("Team 2")
        self._t2_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._t2_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #f87171;")

        comp_layout.addWidget(self._t1_header, 0, 0)
        comp_layout.addWidget(self._vs_label, 0, 1)
        comp_layout.addWidget(self._t2_header, 0, 2)

        # Stat rows
        self._stat_labels: List[tuple] = []
        stat_names = [
            "Avg PPG", "Avg 3PT%", "Avg FG%", "Avg APG",
            "Avg RPG", "Avg +/-", "Team Score", "Win Prob",
        ]
        for i, name in enumerate(stat_names):
            t1_val = QLabel("—")
            t1_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t1_val.setStyleSheet("color: #e2e8f0; font-size: 13px;")
            cat = QLabel(name)
            cat.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cat.setStyleSheet("color: #94a3b8; font-size: 11px;")
            t2_val = QLabel("—")
            t2_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t2_val.setStyleSheet("color: #e2e8f0; font-size: 13px;")

            comp_layout.addWidget(t1_val, i + 1, 0)
            comp_layout.addWidget(cat, i + 1, 1)
            comp_layout.addWidget(t2_val, i + 1, 2)
            self._stat_labels.append((t1_val, cat, t2_val))

        self._comparison.setLayout(comp_layout)
        layout.addWidget(self._comparison)

        # Insights
        self._insights = QLabel("")
        self._insights.setWordWrap(True)
        self._insights.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 4px; "
            "background: #172333; border-radius: 4px;"
        )
        layout.addWidget(self._insights)

        layout.addStretch()
        self.setLayout(layout)
        self._t1_stats: List[Dict] = []
        self._t2_stats: List[Dict] = []
        self._workers: List[_StatsWorker] = []

    def init_data(self) -> None:
        self._team1_picker.load_players()
        self._team2_picker.load_players()

    def _load_matchup(self, team1_key: str, team2_key: str) -> None:
        """Load two ASG teams and their odds, then auto-analyze."""
        t1 = _find_player_ids(ASG_TEAMS.get(team1_key, []))
        t2 = _find_player_ids(ASG_TEAMS.get(team2_key, []))
        if t1:
            self._team1_picker.set_player_ids(t1)
        if t2:
            self._team2_picker.set_player_ids(t2)
        self._team1_odds.setValue(ASG_TEAM_ODDS.get(team1_key, 0))
        self._team2_odds.setValue(ASG_TEAM_ODDS.get(team2_key, 0))
        self._analyze()

    def _prefill_game1(self) -> None:
        """Game 1: USA Stars vs Team World."""
        self._load_matchup("USA Stars (Young)", "Team World")

    def _prefill_stripes_vs_world(self) -> None:
        """USA Stripes vs Team World."""
        self._load_matchup("USA Stripes (Veterans)", "Team World")

    def _prefill_stars_vs_stripes(self) -> None:
        """USA Stars vs USA Stripes."""
        self._load_matchup("USA Stars (Young)", "USA Stripes (Veterans)")

    def _analyze(self) -> None:
        t1_ids = self._team1_picker.selected_ids()
        t2_ids = self._team2_picker.selected_ids()
        if not t1_ids or not t2_ids:
            self._insights.setText("Add players to both teams to analyze.")
            return

        recent = self._recent_spin.value()
        self._pending = 2

        w1 = _StatsWorker(t1_ids, recent)
        w1.finished.connect(self._on_t1_loaded)
        w1.start()

        w2 = _StatsWorker(t2_ids, recent)
        w2.finished.connect(self._on_t2_loaded)
        w2.start()

        self._workers = [w1, w2]
        self._insights.setText("Loading stats...")

    def _on_t1_loaded(self, stats: List[Dict]) -> None:
        self._t1_stats = stats
        self._pending -= 1
        if self._pending == 0:
            self._update_comparison()

    def _on_t2_loaded(self, stats: List[Dict]) -> None:
        self._t2_stats = stats
        self._pending -= 1
        if self._pending == 0:
            self._update_comparison()

    def _update_comparison(self) -> None:
        def _team_avg(stats: List[Dict], key: str) -> float:
            if not stats:
                return 0
            return sum(s.get(key, 0) for s in stats) / len(stats)

        def _team_score(stats: List[Dict]) -> float:
            """Composite team strength score."""
            if not stats:
                return 0
            ppg = _team_avg(stats, "ppg")
            fg3 = _team_avg(stats, "fg3_pct")
            fg = _team_avg(stats, "fg_pct")
            apg = _team_avg(stats, "apg")
            rpg = _team_avg(stats, "rpg")
            pm = _team_avg(stats, "plus_minus")
            return ppg * 2 + fg3 * 0.5 + fg * 0.3 + apg * 1.5 + rpg * 0.8 + pm * 1.0

        t1s = _team_score(self._t1_stats)
        t2s = _team_score(self._t2_stats)
        total = t1s + t2s if (t1s + t2s) > 0 else 1
        t1_prob = t1s / total
        t2_prob = t2s / total

        # Format comparison
        stats_data = [
            ("ppg", ".1f"), ("fg3_pct", ".1f"), ("fg_pct", ".1f"),
            ("apg", ".1f"), ("rpg", ".1f"), ("plus_minus", "+.1f"),
        ]
        for i, (key, fmt) in enumerate(stats_data):
            t1v = _team_avg(self._t1_stats, key)
            t2v = _team_avg(self._t2_stats, key)
            suffix = "%" if "pct" in key else ""
            self._stat_labels[i][0].setText(f"{t1v:{fmt}}{suffix}")
            self._stat_labels[i][2].setText(f"{t2v:{fmt}}{suffix}")
            # Highlight the better team
            if t1v > t2v:
                self._stat_labels[i][0].setStyleSheet("color: #10b981; font-size: 13px; font-weight: bold;")
                self._stat_labels[i][2].setStyleSheet("color: #e2e8f0; font-size: 13px;")
            elif t2v > t1v:
                self._stat_labels[i][2].setStyleSheet("color: #10b981; font-size: 13px; font-weight: bold;")
                self._stat_labels[i][0].setStyleSheet("color: #e2e8f0; font-size: 13px;")
            else:
                self._stat_labels[i][0].setStyleSheet("color: #e2e8f0; font-size: 13px;")
                self._stat_labels[i][2].setStyleSheet("color: #e2e8f0; font-size: 13px;")

        # Team score row
        self._stat_labels[6][0].setText(f"{t1s:.0f}")
        self._stat_labels[6][2].setText(f"{t2s:.0f}")

        # Win prob row
        self._stat_labels[7][0].setText(f"{t1_prob * 100:.1f}%")
        self._stat_labels[7][2].setText(f"{t2_prob * 100:.1f}%")

        if t1_prob > t2_prob:
            self._stat_labels[7][0].setStyleSheet("color: #10b981; font-size: 14px; font-weight: bold;")
            self._stat_labels[7][2].setStyleSheet("color: #f87171; font-size: 14px;")
        else:
            self._stat_labels[7][2].setStyleSheet("color: #10b981; font-size: 14px; font-weight: bold;")
            self._stat_labels[7][0].setStyleSheet("color: #f87171; font-size: 14px;")

        # Odds analysis
        t1_odds = self._team1_odds.value()
        t2_odds = self._team2_odds.value()
        insights_parts = [
            f"Team 1 win probability: {t1_prob * 100:.1f}% "
            f"(fair odds: {implied_to_american(t1_prob)}) | "
            f"Team 2 win probability: {t2_prob * 100:.1f}% "
            f"(fair odds: {implied_to_american(t2_prob)})"
        ]

        if t1_odds != 0:
            t1_implied = american_to_implied(t1_odds)
            t1_edge = (t1_prob - t1_implied) * 100
            insights_parts.append(
                f"Team 1 @ {t1_odds:+d}: implied {t1_implied * 100:.1f}%, "
                f"edge {t1_edge:+.1f}% {'VALUE' if t1_edge > 0 else 'no value'}"
            )
        if t2_odds != 0:
            t2_implied = american_to_implied(t2_odds)
            t2_edge = (t2_prob - t2_implied) * 100
            insights_parts.append(
                f"Team 2 @ {t2_odds:+d}: implied {t2_implied * 100:.1f}%, "
                f"edge {t2_edge:+.1f}% {'VALUE' if t2_edge > 0 else 'no value'}"
            )

        self._insights.setText(" | ".join(insights_parts))


# ════════════════════════════════════════════════════════════════════════
#  Main All-Star View (with sub-tabs)
# ════════════════════════════════════════════════════════════════════════

class AllStarView(QWidget):
    """All-Star Weekend Betting Helper -- main tab widget."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title_row = QHBoxLayout()
        title = QLabel("All-Star Weekend Betting Helper")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #f1f5f9;"
        )
        title_row.addWidget(title)
        title_row.addStretch()

        info = QLabel(
            "Select participants for each event, enter odds, "
            "and get model-based value ratings."
        )
        info.setStyleSheet("color: #64748b; font-size: 11px;")
        title_row.addWidget(info)
        layout.addLayout(title_row)

        # Sub-tabs
        self._tabs = QTabWidget()

        self._mvp = MVPPanel()
        self._three_pt = ThreePointPanel()
        self._rising = RisingStarsPanel()
        self._game = AllStarGamePanel()

        self._tabs.addTab(self._mvp, "  ASG MVP  ")
        self._tabs.addTab(self._three_pt, "  3-Point Contest  ")
        self._tabs.addTab(self._rising, "  Rising Stars  ")
        self._tabs.addTab(self._game, "  Game Winner  ")

        layout.addWidget(self._tabs)
        self.setLayout(layout)
        self._initialized = False

    def showEvent(self, event) -> None:
        """Lazy-load player data on first show."""
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            self._mvp.init_data()
            self._three_pt.init_data()
            self._rising.init_data()
            self._game.init_data()
