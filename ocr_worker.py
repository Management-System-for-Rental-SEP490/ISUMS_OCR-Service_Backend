"""Child-process OCR worker for /ocr/cccd/verify parallel execution.

Why a separate module + process pool:
    The Paddle CPU runtime keeps one global MKL/OpenMP thread pool per
    process. Two PaddleOCR instances in the same process share that pool
    and oversubscribe it — measured 45s/image (3× regression) vs 16s
    single-instance. Running each OCR in its own OS process isolates the
    OpenMP pools so two images truly run in parallel on the 28-core CPU.

Thread budget:
    OMP_NUM_THREADS is set BEFORE paddle imports so each worker clamps
    to 14 threads. Two workers * 14 = 28 physical cores, one thread per
    core, no oversubscription. Setting this AFTER import is too late —
    OpenMP reads the env var on first-touch and caches the pool size.

Model load cost:
    Each worker loads ~800MB of model weights on initializer. Pool of 2
    ≈ 1.6GB extra RAM (the main process still keeps its own primary OCR
    for /ocr/cccd, /front, /back, /passport). Cost paid once at startup
    via `initializer=_ensure_init`; subsequent /verify calls reuse the
    live worker.
"""

import os

# MUST precede any paddle/numpy/cv2 import in this module.
_WORKER_THREADS = int(os.getenv("OCR_WORKER_THREADS", "14"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "FLAGS_cpu_math_library_num_threads"):
    os.environ.setdefault(_var, str(_WORKER_THREADS))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import re  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

_ocr = None


def _ensure_init() -> None:
    """Lazy-init PaddleOCR in the worker process. Called once by the pool's
    `initializer` arg; also guards against pool restart / late-arriving
    tasks if someone ever disables the initializer."""
    global _ocr
    if _ocr is not None:
        return
    _ocr = PaddleOCR(
        lang=os.getenv("OCR_LANG", "en"),
        ocr_version=os.getenv("OCR_VERSION", "PP-OCRv4"),
        text_detection_model_name=os.getenv("OCR_DET_MODEL", "PP-OCRv4_mobile_det"),
        text_recognition_model_name=os.getenv("OCR_REC_MODEL", "PP-OCRv4_mobile_rec"),
        device=os.getenv("OCR_DEVICE", "cpu"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=os.getenv("OCR_USE_TEXTLINE_ORI", "false").lower() == "true",
        text_det_limit_side_len=int(os.getenv("OCR_VERIFY_DET_SIDE_LEN", "640")),
        enable_mkldnn=os.getenv("OCR_ENABLE_MKLDNN", "false").lower() == "true",
        cpu_threads=_WORKER_THREADS,
    )


def ocr_lines(contents: bytes, preprocess_edge: int = 800):
    """Decode image bytes → preprocess to `preprocess_edge`px longest edge
    → run PaddleOCR → return list of cleaned text lines.

    Parsing (front/back classification, ID extraction, MRZ parse) stays
    in the main process — it's pure Python and trivial to transport; no
    reason to pickle the heavy paddle result structures across the IPC
    boundary."""
    _ensure_init()
    arr = np.frombuffer(contents, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge != preprocess_edge:
        scale = preprocess_edge / long_edge
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=interp)
    results = _ocr.predict(rgb)
    out = []
    if not results:
        return out
    for res in results:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        for text, _score in zip(texts, scores):
            # Minimal text normalize — same as main.py's normalize_text_basic
            # so downstream parsers see identical input. We duplicate it
            # here (rather than importing from main.py) so this module is
            # safe to load in a spawned child process that shouldn't
            # re-run main's module-level init.
            if not text:
                continue
            t = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
            t = t.replace("|", " ").replace("\\", "/")
            t = re.sub(r"[\u2010-\u2015]", "-", t)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                out.append(t)
    return out
