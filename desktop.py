"""Entry point — PySide6 Desktop GUI launcher."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Initialize DB, start injury monitor, launch PySide6 GUI."""
    # Ensure src package patches nba_api headers on import
    import src  # noqa: F401

    from src.database.migrations import init_db
    logger.info("Initializing database...")
    init_db()

    # Start injury monitor in background
    try:
        from src.notifications.injury_monitor import InjuryMonitor
        monitor = InjuryMonitor()
        monitor.start()
        logger.info("Injury monitor started (5-min polling)")
    except Exception as e:
        logger.warning(f"Injury monitor failed to start: {e}")

    # Launch PySide6 app
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("NBA Game Prediction System")

    window = MainWindow()
    window.show()

    logger.info("Desktop GUI launched")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
