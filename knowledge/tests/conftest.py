import sys
from pathlib import Path

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE_ROOT))
