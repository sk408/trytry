from src.database import db
rows = db.fetch_all("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for r in rows:
    print(r["name"])
