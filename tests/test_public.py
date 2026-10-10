import ast
import re
from pathlib import Path

from srl_search.public import PUBLIC_DOCS, PUBLIC_ROUTES

GUIDE = Path(__file__).resolve().parents[1] / "web" / "llms.txt"
DOOR = Path(__file__).resolve().parents[2] / "sciencemadereadable" / "app" / "server.py"


def door_routes() -> re.Pattern:
    """DATA_ROUTES from the public door, read without importing (and starting) the site's server."""
    for node in ast.walk(ast.parse(DOOR.read_text())):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DATA_ROUTES":
            return re.compile(ast.literal_eval(node.value.args[0]))
    raise AssertionError("DATA_ROUTES not found")


def test_the_door_passes_every_documented_route():
    routes = door_routes()
    for path in PUBLIC_ROUTES + PUBLIC_DOCS:
        example = path.replace("{ref}", "10.1126/science.aan8048")
        assert routes.match("/api/data" + example), example
    assert not routes.match("/api/data/health")


def test_guide_has_its_coverage_slot_and_mentions_every_route():
    text = GUIDE.read_text()
    assert text.count("{{COVERAGE}}") == 1
    for path in PUBLIC_ROUTES + ("/llms.txt", "/docs", "/openapi.json"):
        assert path.split("{")[0].rstrip("/") in text, path
