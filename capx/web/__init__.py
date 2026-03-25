"""CaP-X Interactive Web UI Backend.

This module provides a FastAPI-based web server with WebSocket support
for real-time interactive robot code execution demos.
"""

from capx.web.server import create_app

__all__ = ["create_app"]
