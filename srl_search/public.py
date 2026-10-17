"""What the internet sees of this API: the public door (sciencemadereadable/app/server.py, DATA_ROUTES) passes
these routes under PUBLIC_BASE, plus the guide and the reference. A test keeps the two lists in step."""
PUBLIC_BASE = "https://api.sciencemadereadable.com/api/data"
# routes in the published OpenAPI spec (FastAPI path templates)
PUBLIC_ROUTES = ("/search", "/search/{q}", "/similar/{ref}", "/works/{ref}", "/walk", "/map/place", "/map/regions", "/map/counts",
                 "/topics", "/topics/{q}", "/fields", "/fields/queue", "/fields/suggest",
                 "/authors", "/authors/{ref}", "/works/{ref}/authors", "/works/{ref}/context", "/works/{ref}/links", "/institutions", "/institutions/{ref}")
# the MCP endpoint (POST /mcp here and at the door; srl_search/mcp_server.py)
PUBLIC_MCP = "https://api.sciencemadereadable.com/mcp"
# documentation the door also passes
PUBLIC_DOCS = ("/llms.txt", "/docs", "/openapi.json")
