"""Entry point — PySide6 Desktop GUI launcher."""

import faulthandler
import logging
import sys
import traceback

# Print native crash tracebacks (SIGSEGV / SIGABRT)
faulthandler.enable()

from src.bootstrap import setup_logging, bootstrap, shutdown

setup_logging()

logger = logging.getLogger(__name__)

# Global exception hook — prevents PySide6 from crashing on unhandled
# exceptions in signal slots (which propagate to C++ and terminate).


def _gui_excepthook(exc_type, exc_value, exc_tb):
    """Log unhandled exceptions instead of crashing the app."""
    logger.error(
        "Unhandled exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
    )


sys.excepthook = _gui_excepthook


def main():
    """Show splash, bootstrap shared services, launch PySide6 GUI."""
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("NBA Game Prediction System")

    # Show splash immediately — before any heavy init
    from src.ui.splash import SplashScreen

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Bootstrap with splash status updates
    bootstrap(status_callback=lambda msg, prog=None: (splash.set_status(msg, prog), app.processEvents()))

    # Build main window (no animation running yet, so no stutter)
    splash.set_status("Loading interface...", 0.9)
    app.processEvents()

    from src.ui.main_window import MainWindow

    window = MainWindow()

    # Now start the constellation animation + linger period
    # All heavy work is done, so animation runs uninterrupted
    splash.set_status("Ready", 1.0)
    app.processEvents()
    splash.start_linger(window)

    logger.info("Desktop GUI launched")
    ret = app.exec()

    # ---- Shutdown: reuse the same splash as a closing screen ----
    import time as _time

    splash.show_shutdown()
    app.processEvents()

    t0 = _time.monotonic()
    shutdown()
    remaining = 2.0 - (_time.monotonic() - t0)
    if remaining > 0:
        _time.sleep(remaining)

    splash.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
