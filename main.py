"""Entry point — FastAPI web server on port 8000."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Initialize DB, start injury monitor, launch Uvicorn."""
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

    # Launch FastAPI via Uvicorn
    import uvicorn
    from src.web.app import app

    logger.info("Starting web server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
