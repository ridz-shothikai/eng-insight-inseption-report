# utils/file_handler.py
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


def cleanup_old_files(directory: Union[str, Path], days: int = 7) -> int:
    """
    Delete files older than specified days.
    
    Args:
        directory: Directory to clean
        days: Age threshold in days
    
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0
    
    cutoff_time = time.time() - (days * 86400)
    deleted_count = 0
    
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
        return deleted_count


def ensure_directories(directories: List[Union[str, Path]]) -> None:
    """
    Ensure all directories in the list exist, create if they don't.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")

# -----------------------------
# JSON File Utilities
# -----------------------------
def load_json(file_path: Union[str, Path]) -> Dict:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"JSON file not found: {file_path}")
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return {}

def save_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"JSON saved: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")


# -----------------------------
# Text File Utilities
# -----------------------------
def read_text(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"Text file not found: {file_path}")
        return ""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        return ""

def write_text(content: str, file_path: Union[str, Path]):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Text file saved: {file_path}")
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {e}")
