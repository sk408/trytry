"""Entry point — FastAPI web server on port 8000."""

from src.bootstrap import setup_logging, bootstrap

setup_logging()

import logging

logger = logging.getLogger(__name__)


def main():
    """Initialize shared services, launch Uvicorn."""
    bootstrap()

    import uvicorn
    from src.web.app import app

    logger.info("Starting web server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
