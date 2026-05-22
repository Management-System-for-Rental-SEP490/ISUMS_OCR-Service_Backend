import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "False")

import asyncio
import logging

from mangum import Mangum

from main import app, _start_ocr_pool

logger = logging.getLogger("cccd-ocr-lambda")
_pool_warmed = False


def _ensure_pool_warm() -> None:
    global _pool_warmed
    if _pool_warmed:
        return
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_start_ocr_pool())
        finally:
            loop.close()
        _pool_warmed = True
        logger.info("OCR pool warmed at Lambda init")
    except Exception:
        logger.exception("OCR pool warmup failed at Lambda init (continuing)")


_ensure_pool_warm()

_mangum = Mangum(app, lifespan="off", api_gateway_base_path=None)


def handler(event, context):
    return _mangum(event, context)
