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


# ════════════════════════════════════════════════════════════════════════
#  Scoring models
# ════════════════════════════════════════════════════════════════════════

def compute_mvp_score(stat: Dict) -> float:
    """Score a player for ASG MVP probability.

    MVP is typically won by the highest scorer on the winning team,
    with bonus for highlight plays (dunks, 3s) and efficiency.
    ASG minutes matter — starters get more.

    Weights emphasise scoring volume, efficiency, and star power.
    """
    pts = stat.get("ppg", 0)
    fg3 = stat.get("fg3m_pg", 0)
    ast = stat.get("apg", 0)
    eff = stat.get("fg_pct", 0) / 100
    ft = stat.get("ft_pct", 0) / 100
    mpg = stat.get("mpg", 0)

    # Scoring volume (most important — MVP almost always top scorer)
    score = pts * 2.5
    # 3-point bonus (highlight factor)
    score += fg3 * 3.0
    # Assists (playmaking creates excitement)
    score += ast * 1.5
    # Efficiency bonus
    score += eff * 15
    score += ft * 5
    # Minutes proxy for star status / likely ASG minutes
    score += min(mpg, 36) * 0.5

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
            "MVP is typically the highest scorer on the winning team. "
            "Model weights: scoring volume, 3PT makes, assists, efficiency, and star minutes."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Player picker
        self._picker = PlayerPicker("MVP Candidates", max_players=24)
        layout.addWidget(self._picker)

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

    def _analyze(self) -> None:
        ids = self._picker.selected_ids()
        if not ids:
            self._insights.setText("Add players using the search above to begin analysis.")
            return
        recent = self._recent_spin.value()
        self._worker = _StatsWorker(ids, recent)
        self._worker.finished.connect(self._on_stats_loaded)
        self._worker.start()
        self._insights.setText("Loading stats...")

    def _on_stats_loaded(self, stats: List[Dict]) -> None:
        self._stats = stats
        self._odds_input.set_players(stats)
        self._refresh_table()

    def _refresh_table(self) -> None:
        odds = self._odds_input.get_odds_map()
        self._table.populate(
            self._stats,
            compute_mvp_score,
            lambda s: f"{s['ppg']:.1f} PPG / {s['fg3m_pg']:.1f} 3PM / {s['apg']:.1f} APG",
            odds,
        )
        # Generate insights
        if self._stats:
            top = sorted(self._stats, key=compute_mvp_score, reverse=True)
            best = top[0]
            self._insights.setText(
                f"Top pick: {best['name']} ({best['team_abbr']}) — "
                f"{best['ppg']:.1f} PPG, {best['fg3m_pg']:.1f} 3PM/G, "
                f"{best['fg_pct']:.1f}% FG. "
                f"Historical ASG MVPs average 25+ points with high efficiency."
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

        self._picker = PlayerPicker("3PT Contestants", max_players=8)
        layout.addWidget(self._picker)

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

    def _analyze(self) -> None:
        ids = self._picker.selected_ids()
        if not ids:
            self._insights.setText("Add 3-point contest participants to begin analysis.")
            return
        recent = self._recent_spin.value()
        self._worker = _StatsWorker(ids, recent)
        self._worker.finished.connect(self._on_stats_loaded)
        self._worker.start()
        self._insights.setText("Loading stats...")

    def _on_stats_loaded(self, stats: List[Dict]) -> None:
        self._stats = stats
        self._odds_input.set_players(stats)
        self._refresh_table()

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
    """Rising Stars Tournament analysis panel."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        header = QLabel("Rising Stars Tournament Predictor")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(header)

        desc = QLabel(
            "Ranks young players by overall production: scoring, rebounding, "
            "assists, stocks (steals + blocks), and efficiency. "
            "Tournament format favors versatile scorers."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        self._picker = PlayerPicker("Rising Stars", max_players=24)
        layout.addWidget(self._picker)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Recent games:"))
        self._recent_spin = QSpinBox()
        self._recent_spin.setRange(0, 82)
        self._recent_spin.setValue(15)
        self._recent_spin.setSpecialValueText("Full season")
        filter_row.addWidget(self._recent_spin)
        filter_row.addStretch()

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(100)
        analyze_btn.clicked.connect(self._analyze)
        filter_row.addWidget(analyze_btn)
        layout.addLayout(filter_row)

        odds_group = QGroupBox("Enter Betting Odds (MVP / Top Scorer)")
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

    def _analyze(self) -> None:
        ids = self._picker.selected_ids()
        if not ids:
            self._insights.setText("Add Rising Stars participants to begin analysis.")
            return
        recent = self._recent_spin.value()
        self._worker = _StatsWorker(ids, recent)
        self._worker.finished.connect(self._on_stats_loaded)
        self._worker.start()
        self._insights.setText("Loading stats...")

    def _on_stats_loaded(self, stats: List[Dict]) -> None:
        self._stats = stats
        self._odds_input.set_players(stats)
        self._refresh_table()

    def _refresh_table(self) -> None:
        odds = self._odds_input.get_odds_map()
        self._table.populate(
            self._stats,
            compute_rising_star_score,
            lambda s: (
                f"{s['ppg']:.1f} PPG / {s['rpg']:.1f} RPG / "
                f"{s['apg']:.1f} APG / {s['spg'] + s['bpg']:.1f} STK"
            ),
            odds,
        )
        if self._stats:
            top = sorted(self._stats, key=compute_rising_star_score, reverse=True)
            best = top[0]
            self._insights.setText(
                f"Top pick: {best['name']} ({best['team_abbr']}) — "
                f"{best['ppg']:.1f} PPG, {best['rpg']:.1f} RPG, "
                f"{best['apg']:.1f} APG. "
                f"Versatile producers with high usage tend to dominate shortened games."
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
            "Enter each team's roster below."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Two team pickers side by side
        teams_row = QHBoxLayout()

        team1_box = QGroupBox("Team 1 (e.g. East / Team LeBron)")
        t1_layout = QVBoxLayout()
        self._team1_picker = PlayerPicker("Team 1", max_players=12)
        t1_layout.addWidget(self._team1_picker)
        team1_box.setLayout(t1_layout)

        team2_box = QGroupBox("Team 2 (e.g. West / Team Giannis)")
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
