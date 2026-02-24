"""Prediction and live-odds info panel for the gamecast sidebar."""

import logging
from typing import Optional, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QProgressBar,
)

logger = logging.getLogger(__name__)


class _InfoCard(QFrame):
    """Styled card for a prediction or odds section."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("infoCard")
        self.setStyleSheet("""
            #infoCard {
                background: rgba(20, 30, 45, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(4)

        header = QLabel(title)
        header.setStyleSheet("""
            color: #00e5ff; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase;
            font-family: 'Oswald', sans-serif;
        """)
        self._layout.addWidget(header)

    def add_row(self, label: str, value: str, value_color: str = "#e2e8f0",
                bold: bool = False):
        """Add a label-value row."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        row.addWidget(lbl)

        val = QLabel(value)
        weight = "700" if bold else "500"
        val.setStyleSheet(f"color: {value_color}; font-size: 12px; font-weight: {weight};")
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(val)
        self._layout.addLayout(row)
        return val


class WinProbBar(QWidget):
    """Horizontal bar showing home/away win probability with team colors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self._home_pct = 50.0
        self._home_color = "#3b82f6"
        self._away_color = "#ef4444"
        self._home_abbr = ""
        self._away_abbr = ""

    def set_data(self, home_pct: float, home_color: str, away_color: str,
                 home_abbr: str = "", away_abbr: str = ""):
        self._home_pct = max(1, min(99, home_pct))
        self._home_color = home_color
        self._away_color = away_color
        self._home_abbr = home_abbr
        self._away_abbr = away_abbr
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background Glass Pill
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 20, 30, 200))
        p.drawRoundedRect(0, 0, w, h, h/2.0, h/2.0)

        home_w = w * self._home_pct / 100.0

        # Away bar (right) - drawn full width then overdrawn by home, or drawn as right portion
        away_grad = QLinearGradient(0, 0, 0, h)
        away_grad.setColorAt(0, QColor(self._away_color).lighter(120))
        away_grad.setColorAt(0.4, QColor(self._away_color))
        away_grad.setColorAt(1, QColor(self._away_color).darker(150))
        
        p.setBrush(away_grad)
        p.drawRoundedRect(0, 0, w, h, h/2.0, h/2.0)

        # Home bar (left)
        if home_w > 0:
            # We want only the left part rounded, and right part straight if it's not 100%. 
            # Or just draw a rounded rect and it looks like a pill inside a pill.
            home_grad = QLinearGradient(0, 0, 0, h)
            home_grad.setColorAt(0, QColor(self._home_color).lighter(120))
            home_grad.setColorAt(0.4, QColor(self._home_color))
            home_grad.setColorAt(1, QColor(self._home_color).darker(150))
            
            p.setBrush(home_grad)
            if home_w >= w - 1:
                p.drawRoundedRect(0, 0, home_w, h, h/2.0, h/2.0)
            else:
                # Draw left half rounded, right half straight
                p.drawRoundedRect(0, 0, int(home_w), h, h/2.0, h/2.0)
                p.drawRect(int(h/2.0), 0, int(home_w - h/2.0), h)

        # Labels
        p.setFont(QFont("Oswald", 10, QFont.Weight.Bold))
        
        # Shadow helper
        def draw_text(x, y, text):
            p.setPen(QColor(0, 0, 0, 180))
            p.drawText(int(x+1), int(y+1), text)
            p.setPen(QColor("#ffffff"))
            p.drawText(int(x), int(y), text)

        if home_w > 40:
            draw_text(10, h - 6, f"{self._home_abbr} {self._home_pct:.0f}%")
        away_pct = 100 - self._home_pct
        if w - home_w > 40:
            text = f"{away_pct:.0f}% {self._away_abbr}"
            tw = p.fontMetrics().horizontalAdvance(text)
            draw_text(w - tw - 10, h - 6, text)

        p.end()


class InfoPanelWidget(QWidget):
    """Combined prediction + odds panel for gamecast sidebar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Win probability bar
        self.win_prob_bar = WinProbBar()
        layout.addWidget(self.win_prob_bar)

        # Prediction card
        self.pred_card = _InfoCard("Model Prediction")
        self._pred_spread = self.pred_card.add_row("Spread", "—")
        self._pred_total = self.pred_card.add_row("Total", "—")
        self._pred_home = self.pred_card.add_row("Home", "—", bold=True)
        self._pred_away = self.pred_card.add_row("Away", "—", bold=True)
        self._pred_winner = self.pred_card.add_row("Pick", "—", "#22c55e", bold=True)
        layout.addWidget(self.pred_card)

        # Odds card
        self.odds_card = _InfoCard("Live Odds")
        self._odds_spread = self.odds_card.add_row("Spread", "—")
        self._odds_ou = self.odds_card.add_row("O/U", "—")
        self._odds_home_ml = self.odds_card.add_row("Home ML", "—")
        self._odds_away_ml = self.odds_card.add_row("Away ML", "—")
        self._odds_provider = self.odds_card.add_row("Source", "—", "#64748b")
        layout.addWidget(self.odds_card)

        layout.addStretch()

    def update_prediction(self, pred: Optional[Dict],
                          home_abbr: str = "", away_abbr: str = ""):
        """Update model prediction display."""
        if not pred:
            self._pred_spread.setText("—")
            self._pred_total.setText("—")
            self._pred_home.setText("—")
            self._pred_away.setText("—")
            self._pred_winner.setText("—")
            return

        spread = pred.get("spread", 0)
        total = pred.get("total", 0)
        home_proj = pred.get("home_projected", 0)
        away_proj = pred.get("away_projected", 0)

        spread_text = f"{home_abbr} {spread:+.1f}" if spread != 0 else "PICK"
        self._pred_spread.setText(spread_text)
        self._pred_total.setText(f"{total:.1f}")
        self._pred_home.setText(f"{home_abbr} {home_proj:.1f}")
        self._pred_away.setText(f"{away_abbr} {away_proj:.1f}")

        winner = home_abbr if spread > 0 else away_abbr
        self._pred_winner.setText(winner)
        self._pred_winner.setStyleSheet(
            f"color: #22c55e; font-size: 12px; font-weight: 700;"
        )

    def update_odds(self, odds: Dict):
        """Update live odds display."""
        if not odds:
            return
        self._odds_spread.setText(str(odds.get("spread", "—")))
        ou = odds.get("over_under")
        self._odds_ou.setText(f"{ou}" if ou else "—")
        home_ml = odds.get("home_moneyline")
        away_ml = odds.get("away_moneyline")
        self._odds_home_ml.setText(f"{home_ml:+d}" if home_ml else "—")
        self._odds_away_ml.setText(f"{away_ml:+d}" if away_ml else "—")
        self._odds_provider.setText(odds.get("provider", "—"))

    def update_win_probability(self, home_pct: float,
                                home_color: str = "#3b82f6",
                                away_color: str = "#ef4444",
                                home_abbr: str = "", away_abbr: str = ""):
        self.win_prob_bar.set_data(home_pct, home_color, away_color,
                                   home_abbr, away_abbr)
