import logging

log = logging.getLogger("server")
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)
