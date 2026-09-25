import logging

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@pytest.fixture(autouse=True)
def log_case(request):
    logging.getLogger("test").info("=== %s ===", request.node.nodeid)
