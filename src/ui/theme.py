"""Dark theme QSS stylesheet for PySide6 GUI."""

GLOBAL_STYLESHEET = """
/* ---- Global Dark Theme ---- */
QWidget {
    background-color: #0f1923;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #0f1923;
}

/* ---- Tab Widget ---- */
QTabWidget::pane {
    border: 1px solid #1e293b;
    background: #0f1923;
    border-radius: 4px;
}

QTabBar::tab {
    background: #1e293b;
    color: #94a3b8;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background: #0f1923;
    color: #3b82f6;
    border-bottom: 2px solid #3b82f6;
}

QTabBar::tab:hover {
    background: #334155;
    color: #e2e8f0;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 500;
    min-height: 32px;
}

QPushButton:hover {
    background-color: #2563eb;
}

QPushButton:pressed {
    background-color: #1d4ed8;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}

QPushButton[class="danger"] {
    background-color: #ef4444;
}

QPushButton[class="danger"]:hover {
    background-color: #dc2626;
}

QPushButton[class="outline"] {
    background-color: transparent;
    border: 1px solid #475569;
    color: #94a3b8;
}

QPushButton[class="outline"]:hover {
    border-color: #3b82f6;
    color: #3b82f6;
}

QPushButton[class="success"] {
    background-color: #22c55e;
}

/* ---- Input Fields ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e2e8f0;
    min-height: 32px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3b82f6;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #3b82f6;
    color: #e2e8f0;
}

/* ---- Tables ---- */
QTableWidget, QTableView {
    background-color: #0f1923;
    alternate-background-color: #141e2b;
    gridline-color: #1e293b;
    border: 1px solid #1e293b;
    border-radius: 4px;
    selection-background-color: rgba(59, 130, 246, 0.15);
    selection-color: #e2e8f0;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px 10px;
    border: none;
    border-bottom: 2px solid #334155;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* ---- Scroll Bars ---- */
QScrollBar:vertical {
    background: #0f1923;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #0f1923;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 4px;
    min-width: 32px;
}

/* ---- Labels ---- */
QLabel {
    color: #e2e8f0;
    background: transparent;
}

QLabel[class="header"] {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
}

QLabel[class="subheader"] {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
}

QLabel[class="muted"] {
    color: #64748b;
    font-size: 12px;
}

QLabel[class="accent"] {
    color: #3b82f6;
    font-weight: 700;
}

QLabel[class="success"] {
    color: #22c55e;
}

QLabel[class="danger"] {
    color: #ef4444;
}

QLabel[class="warning"] {
    color: #f59e0b;
}

/* ---- Text Edit (Log Output) ---- */
QTextEdit, QPlainTextEdit {
    background-color: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    color: #94a3b8;
}

/* ---- Group Box ---- */
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px;
    font-weight: 600;
    color: #e2e8f0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #94a3b8;
}

/* ---- Progress Bar ---- */
QProgressBar {
    background-color: #1e293b;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 3px;
}

/* ---- Splitter ---- */
QSplitter::handle {
    background: #334155;
    width: 2px;
    height: 2px;
}

/* ---- Status Bar ---- */
QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    border-top: 1px solid #334155;
    padding: 4px 12px;
    font-size: 12px;
}

/* ---- Tool Tip ---- */
QToolTip {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}

/* ---- Menu ---- */
QMenu {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3b82f6;
    color: white;
}

/* ---- Check Box ---- */
QCheckBox {
    color: #e2e8f0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #475569;
    border-radius: 4px;
    background: #1e293b;
}

QCheckBox::indicator:checked {
    background: #3b82f6;
    border-color: #3b82f6;
}
"""
