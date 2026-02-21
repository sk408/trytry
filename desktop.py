"""Run the desktop PySide6 version of NBA Betting Analytics."""
import multiprocessing
import sys
import traceback

# Required for ProcessPoolExecutor on Windows / PyInstaller builds
multiprocessing.freeze_support()

# On Windows, faulthandler.enable() registers an SEH handler that catches
# non-fatal COM errors from Qt (code 0x8001010d), printing misleading
# "Windows fatal exception" messages.  Only enable on non-Windows.
if sys.platform != "win32":
    import faulthandler
    faulthandler.enable()


def _excepthook(exc_type, exc_value, exc_tb):
    """Global unhandled-exception handler – print full traceback."""
    # Print to stdout so it shows up in the console widget
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stdout)


sys.excepthook = _excepthook

from src.ui.main_window import run_app  # noqa: E402

if __name__ == "__main__":
    run_app()
