import os
import shutil
import zipfile
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = os.path.join("data", "snapshots")
DB_PATH = os.path.join("data", "nba.db")
TUNING_PATH = os.path.join("src", "analytics", "tuning.py")

def _ensure_dir():
    if not os.path.exists(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

def create_snapshot(name_prefix: str = "auto") -> Optional[str]:
    """
    Creates a zip snapshot containing the database and tuning file, along with metadata.
    Returns the path to the created zip file.
    """
    try:
        _ensure_dir()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}.zip"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)
        
        # Try to get latest metrics from backtest cache
        accuracy = None
        skill_score = None
        ats_record = None
        try:
            from src.analytics.backtester import _load_cache
            cached = _load_cache()
            if cached:
                accuracy = cached.get("accuracy")
                if "quality_metrics" in cached:
                    skill_score = cached["quality_metrics"].get("skill_score")
                    ats = cached["quality_metrics"].get("vegas_comparison", {}).get("ats_record", {})
                    if ats:
                        wins = ats.get("wins", 0)
                        losses = ats.get("losses", 0)
                        pushes = ats.get("pushes", 0)
                        total = wins + losses
                        win_pct = (wins / total * 100) if total > 0 else 0
                        ats_record = f"{wins}-{losses}-{pushes} ({win_pct:.1f}%)"
        except Exception as e:
            logger.warning(f"Could not load backtest cache for snapshot metadata: {e}")
            
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "type": name_prefix,
            "metrics": {
                "accuracy": accuracy,
                "skill_score": skill_score,
                "ats_record": ats_record
            }
        }
        
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(DB_PATH):
                zipf.write(DB_PATH, arcname="nba.db")
            if os.path.exists(TUNING_PATH):
                zipf.write(TUNING_PATH, arcname="tuning.py")
            
            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
        logger.info(f"Created snapshot: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to create snapshot: {e}")
        return None

def list_snapshots() -> List[Dict]:
    """Returns a list of snapshots sorted by date (newest first)."""
    _ensure_dir()
    snapshots = []
    
    for filename in os.listdir(SNAPSHOTS_DIR):
        if not filename.endswith(".zip"):
            continue
            
        filepath = os.path.join(SNAPSHOTS_DIR, filename)
        try:
            with zipfile.ZipFile(filepath, 'r') as zipf:
                if "metadata.json" in zipf.namelist():
                    meta_bytes = zipf.read("metadata.json")
                    meta = json.loads(meta_bytes)
                else:
                    # Fallback for old/manual zips without metadata
                    timestamp = os.path.getmtime(filepath)
                    meta = {
                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                        "type": "unknown",
                        "metrics": {}
                    }
                    
                meta["filename"] = filename
                meta["size_mb"] = os.path.getsize(filepath) / (1024 * 1024)
                snapshots.append(meta)
        except Exception as e:
            logger.error(f"Error reading snapshot {filename}: {e}")
            
    # Sort by timestamp descending
    snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return snapshots

def restore_snapshot(filename: str) -> bool:
    """Restores the database and tuning file from the given snapshot."""
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Snapshot not found: {filepath}")
        return False
        
    try:
        from src.database import db
        # Close connections before overwriting
        db.close_all()
        
        with zipfile.ZipFile(filepath, 'r') as zipf:
            # Extract to temp locations first
            if "nba.db" in zipf.namelist():
                zipf.extract("nba.db", path=os.path.join("data", "temp_extract"))
                shutil.move(os.path.join("data", "temp_extract", "nba.db"), DB_PATH)
            
            if "tuning.py" in zipf.namelist():
                zipf.extract("tuning.py", path=os.path.join("data", "temp_extract"))
                shutil.move(os.path.join("data", "temp_extract", "tuning.py"), TUNING_PATH)
                
        # Clean up temp
        if os.path.exists(os.path.join("data", "temp_extract")):
            shutil.rmtree(os.path.join("data", "temp_extract"))
            
        # Reload the DB into memory after replacing
        db.reload_memory()
        
        logger.info(f"Successfully restored snapshot: {filename}")
        
        # Invalidate backtest cache so the UI reflects the restored state's actual evaluation on next run
        from src.analytics.backtester import _get_cache_path
        try:
            cache_path = _get_cache_path()
            if cache_path and os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception:
            pass
            
        return True
    except Exception as e:
        logger.error(f"Failed to restore snapshot {filename}: {e}")
        return False

def delete_snapshot(filename: str) -> bool:
    """Deletes the specified snapshot."""
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"Deleted snapshot: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete snapshot {filename}: {e}")
            return False
    return False
