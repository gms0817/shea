import logging
import os

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log = logging.getLogger("server")
log.setLevel(log_level)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)

if not log.hasHandlers():
    log.addHandler(handler)
