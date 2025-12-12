#!/usr/bin/env python3
"""
Run the LucidPie web application.
Alternative launcher that handles path issues.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lucidpie.web_app import run_server

if __name__ == "__main__":
    import os
    from lucidpie.production_config import HOST, PORT, PRODUCTION
    
    # Get port from environment, command line, or use default
    port = int(os.getenv("PORT", sys.argv[1] if len(sys.argv) > 1 else PORT))
    host = os.getenv("HOST", HOST)
    reload = not PRODUCTION  # Only reload in development
    
    if PRODUCTION:
        print(f"Starting LucidPie in PRODUCTION mode...")
        print(f"Server running on {host}:{port}")
    else:
        print(f"Starting LucidPie Web Server on port {port}...")
        print(f"Open your browser to: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
    print()
    run_server(host=host, port=port, reload=reload)

