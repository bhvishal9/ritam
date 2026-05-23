from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_INDEXED_CHUNKS_FILE = ASSETS_DIR / "indexed_chunks.json"
DEFAULT_DOCS_DIR = ASSETS_DIR / "docs"
DEFAULT_DESTINATION_DIR = BASE_DIR / "dest"
DEFAULT_SQLITE_DB_PATH = BASE_DIR / "dataset_store.db"
