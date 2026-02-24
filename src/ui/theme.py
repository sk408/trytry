"""Modern Broadcast Theme QSS stylesheet for PySide6 GUI."""

GLOBAL_STYLESHEET = """
/* ---- Global Modern Broadcast Theme ---- */
QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Oswald', 'Segoe UI', sans-serif;
    font-size: 14px;
}

QMainWindow {
    background-color: #0b0f19;
}

/* ---- Tab Widget ---- */
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(20, 30, 45, 0.8);
    border-radius: 4px;
}

QTabBar::tab {
    background: rgba(20, 30, 45, 0.8);
    color: #94a3b8;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
}

QTabBar::tab:selected {
    background: rgba(20, 30, 45, 0.95);
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff;
}

QTabBar::tab:hover {
    background: rgba(30, 45, 65, 0.9);
    color: #e2e8f0;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: rgba(30, 45, 65, 0.8);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
    min-height: 32px;
}

QPushButton:hover {
    background-color: rgba(59, 130, 246, 0.8);
    border-color: #00e5ff;
    color: #00e5ff;
}

QPushButton:pressed {
    background-color: rgba(29, 78, 216, 0.9);
}

QPushButton:disabled {
    background-color: rgba(15, 25, 35, 0.5);
    color: #64748b;
    border-color: transparent;
}

QPushButton[class="danger"] {
    background-color: rgba(239, 68, 68, 0.8);
}

QPushButton[class="danger"]:hover {
    background-color: rgba(220, 38, 38, 0.9);
    border-color: #ff4d4f;
    color: #ffffff;
}

QPushButton[class="outline"] {
    background-color: transparent;
    border: 1px solid #475569;
    color: #94a3b8;
}

QPushButton[class="outline"]:hover {
    border-color: #00e5ff;
    color: #00e5ff;
    background-color: rgba(0, 229, 255, 0.1);
}

QPushButton[class="success"] {
    background-color: rgba(34, 197, 94, 0.8);
}

/* ---- Input Fields ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: rgba(20, 30, 45, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 6px 10px;
    color: #e2e8f0;
    min-height: 32px;
    font-weight: 600;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #00e5ff;
    background-color: rgba(20, 30, 45, 0.95);
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #0b0f19;
    border: 1px solid #00e5ff;
    selection-background-color: rgba(59, 130, 246, 0.5);
    color: #e2e8f0;
}

/* ---- Tables ---- */
QTableWidget, QTableView {
    background-color: rgba(11, 15, 25, 0.6);
    alternate-background-color: rgba(20, 30, 45, 0.6);
    gridline-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    selection-background-color: rgba(0, 229, 255, 0.2);
    selection-color: #00e5ff;
    font-family: 'Segoe UI', sans-serif;
}

QHeaderView::section {
    background-color: rgba(20, 30, 45, 0.8);
    color: #94a3b8;
    padding: 6px 10px;
    border: none;
    border-bottom: 2px solid #00e5ff;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
}

/* ---- Scroll Bars ---- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(0, 229, 255, 0.5);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    min-width: 32px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(0, 229, 255, 0.5);
}

/* ---- Labels ---- */
QLabel {
    color: #e2e8f0;
    background: transparent;
}

QLabel[class="header"] {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
}

QLabel[class="subheader"] {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
}

QLabel[class="muted"] {
    color: #64748b;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}

QLabel[class="accent"] {
    color: #00e5ff;
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
    background-color: rgba(11, 15, 25, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 8px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    color: #94a3b8;
}

/* ---- Group Box ---- */
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    margin-top: 12px;
    padding: 16px 12px 12px;
    font-weight: 700;
    color: #ffffff;
    background: rgba(20, 30, 45, 0.4);
    text-transform: uppercase;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #00e5ff;
}

/* ---- Progress Bar ---- */
QProgressBar {
    background-color: rgba(20, 30, 45, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #00e5ff;
    border-radius: 3px;
}

/* ---- Splitter ---- */
QSplitter::handle {
    background: rgba(255, 255, 255, 0.1);
    width: 2px;
    height: 2px;
}

/* ---- Status Bar ---- */
QStatusBar {
    background-color: #0b0f19;
    color: #94a3b8;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 4px 12px;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}

/* ---- Tool Tip ---- */
QToolTip {
    background-color: rgba(20, 30, 45, 0.95);
    color: #ffffff;
    border: 1px solid #00e5ff;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}

/* ---- Menu ---- */
QMenu {
    background-color: rgba(20, 30, 45, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: rgba(0, 229, 255, 0.2);
    color: #00e5ff;
}

/* ---- Check Box ---- */
QCheckBox {
    color: #e2e8f0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    background: rgba(20, 30, 45, 0.8);
}

QCheckBox::indicator:checked {
    background: #00e5ff;
    border-color: #00e5ff;
}
"""

def setup_theme(widget):
    """Apply the theme to a given widget or QApplication, adjusting for OLED mode."""
    from src.config import get as get_setting
    oled_mode = get_setting("oled_mode", False)
    
    stylesheet = GLOBAL_STYLESHEET
    if oled_mode:
        # Replace main dark backgrounds with pure black for OLED
        stylesheet = stylesheet.replace("#0b0f19", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.95)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.6)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.4)", "#000000")
        stylesheet = stylesheet.replace("rgba(11, 15, 25, 0.6)", "#000000")
        stylesheet = stylesheet.replace("rgba(11, 15, 25, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(15, 25, 35, 0.5)", "#000000")
        stylesheet = stylesheet.replace("rgba(30, 45, 65, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(30, 45, 65, 0.9)", "#000000")
        # Replace hardcoded dark gray colors
        stylesheet = stylesheet.replace("#1e293b", "#000000")
        stylesheet = stylesheet.replace("#0f172a", "#000000")

    widget.setStyleSheet(stylesheet)
