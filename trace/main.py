"""
Main Unified Launcher for TRACE.
Loads configuration, initializes the SQLite database, starts the computer vision pipeline,
and serves the enterprise FastAPI REST & Streaming web dashboard on Uvicorn.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uvicorn

from trace.api.routes import create_app
from trace.config import load_config
from trace.pipeline import TracePipeline
from trace.platform.db import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("trace.main")


def main():
    parser = argparse.ArgumentParser(description="TRACE: Trusted Real-time Asset Confidence Engine")
    parser.add_argument("--config", default="config/demo.yaml", help="Path to YAML configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Web host interface")
    parser.add_argument("--port", type=int, default=8000, help="Web port")
    args = parser.parse_args()

    logger.info("Initializing TRACE Asset Confidence Engine...")
    cfg = load_config(args.config)
    db_mgr = get_db("data/trace.db")

    logger.info(f"Loaded profile: {cfg.profile} for site: {cfg.site}")
    pipeline = TracePipeline(cfg, db_mgr)
    pipeline.start()

    app = create_app(pipeline)

    logger.info(f"Starting TRACE Enterprise Console on http://localhost:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by operator...")
    finally:
        pipeline.stop()
        logger.info("TRACE Pipeline halted cleanly.")


if __name__ == "__main__":
    main()
