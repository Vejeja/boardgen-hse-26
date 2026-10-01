import logging
import os

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger("boardgen")

    log_level = os.environ.get("BOARDGEN_LOG_LEVEL", "INFO").upper()
    _logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not _logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        log_dir = os.environ.get("BOARDGEN_LOG_DIR")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "boardgen.log"))
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)

    return _logger
