import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from multi_agent_security.types import AgentMessage


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "agent": getattr(record, "agent", "system"),
            "event": record.getMessage(),
            "data": getattr(record, "data", {}),
        }
        return json.dumps(entry)


def setup_logging(
    level: str = "INFO",
    output_file: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console (human-readable) and optional file (JSON) handlers."""
    logger = logging.getLogger("masr")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(console)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


class RunLogger:
    """Appends AgentMessage entries as JSON lines to a JSONL file."""

    def __init__(self, output_file: str):
        self._path = Path(output_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_message(self, msg: AgentMessage) -> None:
        with self._path.open("a") as f:
            f.write(msg.model_dump_json() + "\n")
