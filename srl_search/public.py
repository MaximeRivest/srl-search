"""What the internet sees of this API: the public door (sciencemadereadable/app/server.py, DATA_ROUTES) passes
these routes under PUBLIC_BASE, plus the guide and the reference. A test keeps the two lists in step."""
PUBLIC_BASE = "https://api.sciencemadereadable.com/api/data"
# routes in the published OpenAPI spec (FastAPI path templates)
PUBLIC_ROUTES = ("/search", "/search/{q}", "/similar/{ref}", "/works/{ref}", "/walk", "/map/place", "/map/regions", "/map/counts")
# documentation the door also passes
PUBLIC_DOCS = ("/llms.txt", "/docs", "/openapi.json")
