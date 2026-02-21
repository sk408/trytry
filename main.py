import multiprocessing
import uvicorn


def main() -> None:
    """Run the FastAPI web server for the mobile-friendly UI."""
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
