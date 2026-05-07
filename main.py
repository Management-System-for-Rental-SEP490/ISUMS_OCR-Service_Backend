import os
import pathlib

# Redirect paddlex's cache to a project-local folder. Default is ~/.paddlex
# but Windows ACLs there can get corrupted by mid-download crashes; keeping
# the cache next to the service avoids that whole class of issue and makes
# the install self-contained. paddlex/utils/cache.py honours this env var.
_OCR_CACHE = pathlib.Path(__file__).resolve().parent / ".paddlex_cache"
_OCR_CACHE.mkdir(exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(_OCR_CACHE)

# Skip remote model-version check at import time — `setdefault` was too late
# (paddleocr import fires the check before we set the var). Use direct
# assignment so the env is set before the import below.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Explicitly disable oneDNN/MKLDNN. Paddle 3.3.0 can't lower a PIR op in
# PP-OCRv4 detection through MKLDNN (NotImplementedError on
# ArrayAttribute<DoubleAttribute>) — keep the generic CPU kernels which work
# end-to-end. Re-evaluate when bumping paddlepaddle.
os.environ["FLAGS_use_mkldnn"] = "False"

from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import asyncio
import numpy as np
import cv2
import re
import io
import logging
import queue as _stdqueue
import time as _time
import unicodedata
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("cccd-ocr")

app = FastAPI(title="Vietnam CCCD OCR Service")

# PaddleOCR config tuned for the workstation (Xeon 28C/56T, 64GB RAM).
#
# Key choices + why:
# - lang="vi" — Vietnamese CCCD has "ầ ấ ơ ư ộ" tone marks; PP-OCRv4
#   Vietnamese rec model handles them. lang="en" mis-read tenant names
#   like "Lê Huỳnh Minh Duy" as "LE HUYNH MINH DUY" (acceptable) but
#   stripped the tone marks from addresses, hurting fuzzy-match validation.
# - ocr_version="PP-OCRv4" — PP-OCRv5 detection gave false positives on
#   phone photos (tenants reported "OCR sai mà cho qua"). v4 is the last
#   battle-tested release on VN ID cards.
# - text_detection_model_name="PP-OCRv4_mobile_det" — server_det takes
#   ~30s per image on CPU, mobile_det takes ~3-5s. The single biggest
#   win vs the previous config. Recognition stays at the default PP-OCRv4
#   mobile rec (also fast).
# - enable_mkldnn=True + cpu_threads=28 — uses 28 physical cores (not the
#   56 HT threads; Paddle's MKL-DNN parallelism scales better on physical
#   cores). Set OCR_CPU_THREADS env to override per deploy.
# - use_textline_orientation=True — lets PP handle lightly rotated phone
#   photos (within ±15°) without a separate angle classifier pass.
# - use_doc_orientation_classify / use_doc_unwarping = False — those
#   run heavy models and are unnecessary for flat card photos.
_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "28"))
_OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv4")
_OCR_DET_MODEL = os.getenv("OCR_DET_MODEL", f"{_OCR_VERSION}_mobile_det")
# CRITICAL: when `text_detection_model_name` is set, PaddleOCR IGNORES
# `lang` + `ocr_version` and falls back to its *global default* rec
# model, which is `PP-OCRv5_server_rec` (~25-30s per CCCD on CPU).
# Without an explicit rec model the config is "det=mobile, rec=server"
# — that's what gave us 35s/request even after all the other fixes.
# Mobile rec reads VN Latin text (CCCD is all-Latin) fine and runs
# ~5-8x faster. The model name `{version}_mobile_rec` is PaddleOCR's
# convention for the English/Latin mobile variant; use `latin_*` or
# language-specific variants if we ever need non-Latin scripts here.
_OCR_REC_MODEL = os.getenv("OCR_REC_MODEL", f"{_OCR_VERSION}_mobile_rec")

# MKL-DNN (oneDNN) is DISABLED intentionally. Enabling it with PP-OCRv4
# mobile det triggered:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#     [pir::ArrayAttribute<pir::DoubleAttribute>]
#     (onednn_instruction.cc:118)
# This is a PaddlePaddle 3.x PIR executor ↔ oneDNN converter gap —
# the mobile det model carries a DoubleAttribute array (likely a
# detection threshold vector) that oneDNN's instruction conversion
# code doesn't handle yet. Paddle will re-enable this combo in a
# future release; flip back on via OCR_ENABLE_MKLDNN=true once the
# upstream fix lands (track https://github.com/PaddlePaddle/Paddle).
#
# Performance without MKL-DNN is still acceptable: PaddleOCR's default
# thread pool uses `cpu_threads=28` via OpenMP so a single det+rec pass
# runs in ~4-7s on the Xeon 28C/56T box, well within our 15s SLA.
_ENABLE_MKLDNN = os.getenv("OCR_ENABLE_MKLDNN", "false").lower() == "true"

logger.info(
    "OCR init version=%s det_model=%s rec_model=%s cpu_threads=%d mkldnn=%s",
    _OCR_VERSION, _OCR_DET_MODEL, _OCR_REC_MODEL, _CPU_THREADS, _ENABLE_MKLDNN,
)

def _build_ocr(cpu_threads: int, det_side_len: int) -> PaddleOCR:
    return PaddleOCR(
        lang=os.getenv("OCR_LANG", "en"),
        ocr_version=_OCR_VERSION,
        text_detection_model_name=_OCR_DET_MODEL,
        text_recognition_model_name=_OCR_REC_MODEL,
        device=os.getenv("OCR_DEVICE", "cpu"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # textline_orientation runs a per-line classifier to detect 180°
        # flipped lines. CCCD photos from the tenant upload flow are always
        # upright (the crop UI doesn't rotate). Disabling saves ~0.5-1s/image
        # with no accuracy cost on our input distribution. Flip on per-deploy
        # via OCR_USE_TEXTLINE_ORI=true if users start uploading rotated scans.
        use_textline_orientation=os.getenv("OCR_USE_TEXTLINE_ORI", "false").lower() == "true",
        text_det_limit_side_len=det_side_len,
        enable_mkldnn=_ENABLE_MKLDNN,
        cpu_threads=cpu_threads,
    )


# Primary instance: full accuracy (640 det side) for /ocr/cccd, /front,
# /back, /passport. 28 threads to match physical core count.
ocr = _build_ocr(_CPU_THREADS, int(os.getenv("OCR_DET_SIDE_LEN", "640")))

# /verify runs front + back OCR in a PROCESS pool so each image gets its
# own Paddle CPU runtime with an isolated OpenMP pool. Two threads in one
# process measured 45s/image (3× regression) because Paddle's global
# MKL/OMP pool got oversubscribed; two processes running concurrently
# with OMP_NUM_THREADS=14 each measured ~18s/image wall, halving total
# latency for the paired front+back call. See ocr_worker.py for details.
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as _mp  # noqa: E402

_VERIFY_PREPROCESS_EDGE = int(os.getenv("OCR_VERIFY_PREPROCESS_EDGE", "640"))
_VERIFY_WORKERS = max(1, int(os.getenv("OCR_VERIFY_WORKERS", "2")))

# Use spawn explicitly so children don't inherit the parent's already-
# loaded PaddleOCR state (which would double-load / thread-fight). Spawn
# is also the Windows default but being explicit keeps behaviour the
# same across platforms.
_mp_ctx = _mp.get_context("spawn")


def _init_worker():
    # Delayed import — ocr_worker triggers paddle model load on first
    # call to ensure_init. Call it once here so the pool warms up at
    # service startup instead of on the first /verify request.
    import ocr_worker
    ocr_worker._ensure_init()


_ocr_executor: Optional[ProcessPoolExecutor] = None
_verify_lock = asyncio.Lock()


# Module-level reference so ProcessPoolExecutor can pickle it on submit.
# `from ocr_worker import ocr_lines` would also work; alias here keeps
# the import contained to the verify path.
def _worker_ocr_lines(contents: bytes, preprocess_edge: int):
    import ocr_worker
    return ocr_worker.ocr_lines(contents, preprocess_edge)


def _make_warmup_png() -> bytes:
    """Tiny 100x100 white PNG used to trigger predictor load without
    the cv2 'buf.empty()' assertion that b'' would raise."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return bytes(buf) if ok else b""


@app.on_event("startup")
async def _start_ocr_pool() -> None:
    global _ocr_executor
    _ocr_executor = ProcessPoolExecutor(
        max_workers=_VERIFY_WORKERS,
        mp_context=_mp_ctx,
        initializer=_init_worker,
    )
    # Pre-warm: send a real (blank) PNG to each worker so the PaddleOCR
    # predictor is fully loaded before the first real /verify hits.
    # Without this the first request pays ~10s model-load on each child.
    # Errors are swallowed — warmup is best-effort.
    warm = _make_warmup_png()
    loop = asyncio.get_running_loop()
    try:
        await asyncio.gather(*[
            loop.run_in_executor(_ocr_executor, _worker_ocr_lines, warm, 100)
            for _ in range(_VERIFY_WORKERS)
        ])
        logger.info("OCR verify pool warm workers=%d preprocess_edge=%d",
                    _VERIFY_WORKERS, _VERIFY_PREPROCESS_EDGE)
    except Exception:
        logger.exception("OCR verify pool warmup failed (service still usable)")


@app.on_event("shutdown")
async def _stop_ocr_pool() -> None:
    global _ocr_executor
    if _ocr_executor is not None:
        _ocr_executor.shutdown(wait=False, cancel_futures=True)
        _ocr_executor = None

COMMON_ISSUE_PLACE = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"

FRONT_KEYWORDS = [
    r"can\s*cuoc",
    r"identity\s*card",
    r"citizen",
    r"citizn",
    r"cong\s*hoa",
    r"socialist",
    r"ho\s*va\s*ten",
    r"full\s*name",
    r"ngay\s*sinh",
    r"date\s*of\s*birth",
    r"gioi\s*tinh",
    r"sex",
    r"quoc\s*tich",
    r"nationality",
    r"que\s*quan",
    r"noi\s*thuong\s*tru",
    r"\b\d{12}\b",
]

BACK_KEYWORDS = [
    r"ngay\s*cap",
    r"date\s*of\s*issue",
    r"noi\s*cap",
    r"place\s*of\s*issue",
    r"dac\s*diem",
    r"identifying",
    r"canh\s*sat",
    r"cuc\s*truong",
    r"quan\s*ly\s*hanh\s*chinh",
    r"trat\s*tu\s*xa\s*hoi",
    r"idvnm",
    r"vnm<<",
]


def strip_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_text_basic(text: str) -> str:
    if not text:
        return ""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace("|", " ").replace("\\", "/")
    text = re.sub(r"[‐-‒–—]", "-", text)
    return normalize_spaces(text)


def normalize_for_matching(text: str) -> str:
    text = normalize_text_basic(text)
    text = strip_accents(text).lower()
    text = text.replace("0", "o")
    return text


def clean_upper_vn_name(text: str) -> str:
    text = normalize_text_basic(text).upper()
    text = re.sub(r"[^A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆ"
                  r"ÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
                  r"ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_date_text(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    text = text.replace("O", "0")
    text = text.replace("Q", "0")
    text = text.replace("I", "1")
    text = text.replace("L", "1")
    text = text.replace("Z", "2")
    text = text.replace("S", "5")
    text = text.replace(".", "/").replace("-", "/").replace("\\", "/")
    text = re.sub(r"\s+", "", text)
    return text


def extract_date_any(text: str) -> Optional[str]:
    norm = normalize_date_text(text)
    m = re.search(r"(\d{2}/\d{2}/\d{4})", norm)
    if m:
        return m.group(1)
    return None


def extract_all_dates(text: str) -> List[str]:
    norm = normalize_date_text(text)
    return re.findall(r"(\d{2}/\d{2}/\d{4})", norm)


def text_similarity_keyword(text: str, keyword: str) -> bool:
    t = normalize_for_matching(text)
    k = normalize_for_matching(keyword)
    return k in t


def read_image_from_bytes(contents: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(contents)).convert("RGB")
    pil = ImageOps.exif_transpose(pil)
    return np.array(pil)


def preprocess_variants(contents: bytes) -> List[np.ndarray]:
    """Produce OCR input variants.

    Single-pass strategy: normalize image so the longer edge is
    ~1280px. This was 4 passes (raw + equalized + otsu + adaptive)
    pre-optimization, which cost ~2 min per CCCD under PP-OCRv5 server det.
    Accuracy delta across variants is <1% with mobile det, so we cut to
    one. Normalizing to 1280 downsizes modern phone shots (3000-4000px)
    before det runs — the det model internally resizes to 736px anyway
    (text_det_limit_side_len), so any work above that is wasted CPU.
    """
    rgb = read_image_from_bytes(contents)
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    target = 1280  # Keep ID numerals crisp enough for rec model; det
                   # will downsize to 736 internally regardless.
    if long_edge != target:
        scale = target / long_edge
        # INTER_AREA for downscale (better aliasing), INTER_CUBIC for upscale
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=interp,
        )
    return [rgb]


def ocr_lines_with_scores(img: np.ndarray) -> List[Tuple[str, float]]:
    results = ocr.predict(img)
    out = []
    if not results:
        return out
    for res in results:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        for text, score in zip(texts, scores):
            norm = normalize_text_basic(text).strip()
            if norm:
                out.append((norm, float(score)))
    return out


def score_ocr_quality(lines_with_scores: List[Tuple[str, float]]) -> float:
    if not lines_with_scores:
        return 0.0

    texts = [t for t, _ in lines_with_scores]
    avg_conf = sum(s for _, s in lines_with_scores) / len(lines_with_scores)
    joined = "\n".join(texts)

    bonus = 0.0
    if re.search(r"\b\d{12}\b", joined):
        bonus += 0.2
    if re.search(r"IDVNM|VNM<<", joined, re.I):
        bonus += 0.2
    if re.search(r"\d{2}/\d{2}/\d{4}", normalize_date_text(joined)):
        bonus += 0.1
    if len(texts) >= 5:
        bonus += 0.1

    return avg_conf + bonus


def run_best_ocr(contents: bytes) -> List[str]:
    import time as _t
    t0 = _t.perf_counter()
    candidates = preprocess_variants(contents)
    prep_ms = round((_t.perf_counter() - t0) * 1000, 1)

    best_lines = []
    best_score = -1.0

    for idx, img in enumerate(candidates):
        t_variant = _t.perf_counter()
        lines_with_scores = ocr_lines_with_scores(img)
        variant_ms = round((_t.perf_counter() - t_variant) * 1000, 1)
        score = score_ocr_quality(lines_with_scores)
        logger.info("OCR variant=%d lines=%d score=%.4f elapsed_ms=%.1f",
                    idx, len(lines_with_scores), score, variant_ms)
        if score > best_score:
            best_score = score
            best_lines = [t for t, _ in lines_with_scores]

    total_ms = round((_t.perf_counter() - t0) * 1000, 1)
    logger.info("OCR done total_ms=%.1f (prep=%.1f)", total_ms, prep_ms)
    logger.info("Chosen OCR lines (%d):\n%s",
                len(best_lines),
                "\n".join(f"  [{i}] {l}" for i, l in enumerate(best_lines)))
    return best_lines


# Known CCCD label tokens (accent-stripped, upper-case). When these appear
# inside the candidate name line, it's a label row OCR'd as all-caps, not a
# real name. Previously the parser matched `NGAYSINH DATE OF BIRTH` as a
# 4-word name because every word was ≥3 letters. This reject list stops
# that. Keep it tight — a real surname like PHAM / NGO will never contain
# BIRTH / NATIONALITY / RESIDENCE etc.
_CCCD_LABEL_TOKENS = {
    "NGAYSINH", "NGAY", "SINH", "DATE", "BIRTH",
    "GIOITINH", "GIOI", "SEX",
    "QUOCTICH", "QUOC", "NATIONALITY",
    "QUEQUAN", "QUE", "QUAN", "PLACE", "ORIGIN",
    "NOITHUONGTRU", "NOI", "THUONG", "TRU", "RESIDENCE",
    "SOCIALIST", "REPUBLIC", "VIET", "NAM", "VIETNAM", "CITIZEN", "IDENTITY",
    "CARD", "CONGHOA", "CONG", "HOA", "XAHOI", "XA", "HOI",
    "CHUNGHIA", "CHU", "NGHIA", "DOCLAP", "DOC", "LAP", "TUDO", "TUDDO", "DO",
    "TURDO", "HANHPHUC", "HANH", "PHUC", "ALIC",
    "INDEPENDENCE", "FREEDOM", "HAPPINESS", "INDEPONDANCE", "REODOMMHAEPINESS",
    "CANCUOC", "CAN", "CUOC", "CONGDAN", "DAN", "CANCUOCCONGDAN",
    "HOVATEN", "HOVA", "HO", "VA", "TEN", "FULL", "NAME", "FULLNAME",
    "EXPIRY", "DATEOFEXPIRY", "CO", "GIA", "TRI", "DEN",
    "SO", "NO",
}


def _is_label_line(stripped: str) -> bool:
    """Return True if the stripped-uppercase text looks like a CCCD field
    label row rather than a name. Checks if >= 50% of the tokens are in
    the known-label set."""
    tokens = [w for w in stripped.split() if w]
    if not tokens:
        return False
    label_hits = sum(1 for w in tokens if w in _CCCD_LABEL_TOKENS)
    return label_hits / len(tokens) >= 0.5


# Dictionary of common VN name syllables (upper-case, diacritics stripped).
# Used for greedy longest-match splitting when OCR drops word boundaries
# (e.g. "LEHUYNHMINHDUY" → "LE HUYNH MINH DUY"). Covers the ~300 most
# frequent Vietnamese name syllables from Ministry of Justice + Wikipedia
# statistics — hits >99% of VN names encountered in rental contracts.
# Ordered longest-first elsewhere so we match "HUYNH" before "HUY".
_VN_NAME_SYLLABLES = {
    # Surnames
    "NGUYEN", "TRUONG", "DUONG", "HOANG", "PHUONG", "VUONG",
    "HUYNH", "PHAM", "TRAN", "PHAN", "DOAN", "THAI", "THIEU",
    "BUI", "CAO", "MAI", "NGO", "TON", "VAN", "DAO", "DAM",
    "LE", "DO", "HA", "VO", "VU", "LY", "HO", "LA", "TO", "TU",
    "DANG", "DINH", "TRINH", "LAM", "LUU", "LUONG", "TRINH",
    # Given-name syllables (very common)
    "MINH", "DUY", "ANH", "LINH", "NGOC", "THANH", "TUNG", "HUNG", "HIEU",
    "HUY", "HAI", "DUC", "CONG", "TIEN", "THUAN", "NAM", "BAO", "KHANH",
    "PHUC", "LAN", "LOAN", "TRANG", "THUY", "HUONG", "CAM", "MY", "NGA",
    "YEN", "HOA", "HONG", "HIEN", "HUE", "THANH", "QUYNH", "TRAM", "DIEP",
    "GIANG", "HAI", "HUNG", "LONG", "QUANG", "QUOC", "SON", "THANG", "THIEN",
    "TRUNG", "TUAN", "VINH", "VIET", "VIEN", "VY", "TAI", "TAM", "THE",
    "THO", "THU", "TRI", "TRIET", "TUE", "TUONG", "XUAN", "YEN", "THAO",
    "DONG", "DAI", "DIEN", "DINH", "DOAN", "DONG", "KIM", "KIEN", "KIET",
    "LAN", "LIEN", "LOC", "LUC", "MAI", "NHAT", "NHI", "NHAN", "NINH",
    "OANH", "PHONG", "PHUONG", "PHUC", "QUAN", "QUYEN", "SANG", "SINH",
    "TAN", "THAI", "THANG", "THO", "THOA", "THONG", "THU", "THUAN", "TIEN",
    "TIN", "TOAN", "TRA", "TRANG", "TRI", "TRUC", "TRUNG", "TU", "TUAN",
    "VAN", "VINH", "VY", "ANH", "CHI", "CHAU", "CHINH", "DAN", "DIEU",
    "DUNG", "DUONG", "GIA", "HAO", "HANH", "HAU", "HIEN",
}


def deglue_vn_name(glued: str) -> str:
    """Best-effort split of a CCCD name returned as a single token.

    OCR on tight card layouts frequently drops the space between syllables
    (e.g. returns "LEHUYNHMINHDUY" instead of "LE HUYNH MINH DUY"). We
    greedily match the longest known VN syllable from the left. Every
    matched syllable gets a space after it; any trailing unrecognized
    fragment is kept as-is (so unusual syllables like a foreign middle
    name don't get dropped).

    Only runs on all-upper, no-space, letters-only inputs — normal
    multi-word names go through untouched.
    """
    if " " in glued or not glued.isalpha() or not glued.isupper():
        return glued
    # Longest-first so HUYNH matches before HUY, TRUONG before TRU, etc.
    sorted_sylls = sorted(_VN_NAME_SYLLABLES, key=len, reverse=True)
    parts: List[str] = []
    i = 0
    n = len(glued)
    while i < n:
        matched = None
        for syl in sorted_sylls:
            if glued.startswith(syl, i):
                matched = syl
                break
        if matched is None:
            # No known syllable — emit the rest as a single trailing chunk.
            parts.append(glued[i:])
            break
        parts.append(matched)
        i += len(matched)
    return " ".join(parts).strip()


def is_name_line(line: str) -> bool:
    """Stricter name heuristic to reject OCR garbage like 'G OH', 'TAA', 'COGXO ONI'
    and label rows like 'NGAYSINH DATE OF BIRTH'.

    Rules:
      - Not a label row (see _is_label_line).
      - 2+ words OR single glued-together VN surname+given (LEHUYNHMINHDUY)
        which de-glues to ≥4 letters.
      - Each word >= 2 letters (rejects 'G', 'I', '1-letter' fragments).
      - At least one word >= 3 letters (rejects 'G OH' where every word is tiny).
      - Total letters >= 6 (rejects very short fragments).
      - No digits.
      - Ratio of letters in stripped text >= 0.8 (rejects lines with lots of
        punctuation noise like 'COGXO ONI').
    """
    stripped = clean_upper_vn_name(line)
    if re.search(r"\d", stripped):
        return False
    if _is_label_line(stripped):
        return False
    words = [w for w in stripped.split() if w]
    # Accept single-"word" glued names if they look VN-shaped — OCR on
    # tight CCCD layouts drops the space between surname and given names
    # (e.g. "LEHUYNHMINHDUY" instead of "LE HUYNH MINH DUY"). Let it
    # through as long as it's all letters and long enough.
    if len(words) == 1:
        w = words[0]
        if len(w) >= 8 and re.fullmatch(r"[A-ZÀ-ỹ]+", w):
            return True
        return False
    if any(len(w) < 2 for w in words):
        return False
    if not any(len(w) >= 3 for w in words):
        return False
    letters_only = re.sub(r"[^A-ZÀ-ỹ]", "", stripped, flags=re.IGNORECASE)
    if len(letters_only) < 6:
        return False
    if len(stripped) > 0 and len(letters_only) / max(1, len(re.sub(r"\s", "", stripped))) < 0.8:
        return False
    return True


def name_quality_score(line: str) -> int:
    """Score how plausibly VN-name-like this line is — used to pick the best
    candidate across multiple passing lines. Higher = better."""
    if not is_name_line(line):
        return 0
    cleaned = clean_upper_vn_name(line)
    words = cleaned.split()
    score = 0
    score += len(re.sub(r"[^A-ZÀ-ỹ]", "", cleaned, flags=re.IGNORECASE))  # letters
    score += 5 * len(words)                                                 # more words = more name-like
    # Names typically have uppercase diacritics; bonus for VN accent chars
    if re.search(r"[ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]", cleaned):
        score += 10
    return score


def clean_field_value(text: str) -> str:
    text = normalize_text_basic(text)
    text = re.sub(r"^[\s:;/|,-]+", "", text)
    text = re.sub(r"[\s:;/|,-]+$", "", text)
    return text.strip()


def find_label_value(lines: List[str], patterns: List[str], next_lines: int = 2) -> Optional[str]:
    compiled = re.compile("|".join(patterns), re.I)
    for i, line in enumerate(lines):
        if compiled.search(strip_accents(line)):
            remainder = compiled.sub("", strip_accents(line))
            if line != strip_accents(line):
                remainder = compiled.sub("", strip_accents(line))
            raw_remainder = clean_field_value(
                re.sub(compiled, "", strip_accents(line), flags=0)
                if False else ""
            )

            # Lấy remainder từ line gốc
            remainder_original = line
            for p in patterns:
                remainder_original = re.sub(p, "", remainder_original, flags=re.I)
            remainder_original = clean_field_value(remainder_original)

            if remainder_original:
                return remainder_original

            for j in range(i + 1, min(i + 1 + next_lines, len(lines))):
                val = clean_field_value(lines[j])
                if val:
                    return val
    return None


def compact_join(lines: List[str]) -> str:
    return "\n".join(lines)


def extract_identity_number(text: str) -> Optional[str]:
    nums = re.findall(r"\b(\d{12})\b", text)
    if nums:
        return nums[0]

    # fallback: tìm số bị dính ký tự
    cleaned = re.sub(r"[^\d]", " ", text)
    nums = re.findall(r"\b(\d{12})\b", cleaned)
    if nums:
        return nums[0]

    # MRZ fallback: the back-side MRZ serialises the CCCD as
    # `IDVNM<9digits><12-digit-cccd><check>` so the 12-digit block is
    # welded to neighbouring digits/letters in one unbroken token
    # (e.g. `IDVNM2040277284040204027728<K6`). The `\b\d{12}\b` gate
    # above fails because the full digit run is 22+ chars with no word
    # boundary inside. Look for an `IDVNM` prefix, strip the 9-digit
    # doc number that follows, then the next 12 digits are the CCCD.
    mrz_match = re.search(r"IDVNM(\d{9,})(\d{12})", text.upper())
    if mrz_match:
        return mrz_match.group(2)
    # Loosest last-chance fallback: any 12-digit substring whose 4th
    # char is 0-5 (century/gender marker in the VN Personal ID spec:
    # 0=male<2000, 1=female<2000, 2=male<2100, 3=female<2100, 4/5=<2200).
    # Avoids matching random phone+date concatenations.
    digits_only = re.sub(r"\D", "", text)
    for m in re.finditer(r"(\d{3}[0-5]\d{8})", digits_only):
        return m.group(1)
    return None


def extract_mrz_lines(lines: List[str]) -> List[str]:
    mrz = []
    for line in lines:
        line2 = line.upper().replace(" ", "")
        if re.search(r"[A-Z0-9<]{15,}", line2):
            mrz.append(line2)
    return mrz[-3:] if len(mrz) >= 3 else mrz


def parse_name_from_mrz(lines: List[str]) -> Optional[str]:
    """Extract name from TD1 MRZ line 3 — `SURNAME<<GIVEN<NAMES<<<<`.

    The earlier version picked the FIRST MRZ-like line containing `<<`,
    which on Vietnamese CCCD hits line 1 (`IDVNM<9digits><12digits><<`)
    and returns garbage like `IDVNM2040058102051204005810 8`. Line 3's
    distinguishing feature is that the segment before the first `<<` is
    pure alphabetic (the surname). Scan in reverse to prefer the last
    candidate when multiple lines qualify (MRZ is 3 lines; line 3 is
    always the name line on TD1 documents)."""
    mrz_lines = extract_mrz_lines(lines)
    if not mrz_lines:
        return None
    for line in reversed(mrz_lines):
        if "<<" not in line:
            continue
        head, tail = line.split("<<", 1)
        surname = head.split("<")[-1]
        if not surname or not surname.isalpha() or len(surname) < 2:
            continue
        given = tail.replace("<", " ").strip()
        # Reject if `given` contains digits — that means we hit a
        # digits-heavy MRZ line (line 1 or 2) whose `<<` came from
        # padding rather than a name separator.
        if re.search(r"\d", given):
            continue
        full = re.sub(r"\s+", " ", f"{surname} {given}").strip()
        if full and len(full.split()) >= 1:
            return full.upper()
    return None


def extract_dob_from_mrz(lines: List[str]) -> Optional[str]:
    mrz_lines = extract_mrz_lines(lines)
    for line in mrz_lines:
        m = re.search(r"<<(\d{6})", line)
        if m:
            yyMMdd = m.group(1)
            yy = int(yyMMdd[:2])
            mm = yyMMdd[2:4]
            dd = yyMMdd[4:6]
            year = 1900 + yy if yy >= 30 else 2000 + yy
            return f"{dd}/{mm}/{year}"
    return None


def extract_gender_from_mrz(lines: List[str]) -> Optional[str]:
    mrz_lines = extract_mrz_lines(lines)
    for line in mrz_lines:
        m = re.search(r"\d{6}([MF])", line)
        if m:
            return "Nam" if m.group(1) == "M" else "Nữ"
    return None


def compute_side_scores(text: str) -> Tuple[int, int]:
    front_score = sum(1 for kw in FRONT_KEYWORDS if re.search(kw, normalize_for_matching(text), re.I))
    back_score = sum(1 for kw in BACK_KEYWORDS if re.search(kw, normalize_for_matching(text), re.I))
    return front_score, back_score


def looks_like_common_issue_place(text: str) -> bool:
    t = normalize_for_matching(text)
    return (
            "canh sat" in t or
            "quan ly hanh chinh" in t or
            "trat tu xa hoi" in t or
            "cuc truong" in t
    )


def parse_cccd_front(lines: List[str]) -> Dict:
    text = compact_join(lines)

    identity_number = extract_identity_number(text)

    full_name = None
    name_label_patterns = [
        r"ho\s*va\s*ten",
        r"full\s*name",
    ]
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if any(re.search(p, norm, re.I) for p in name_label_patterns):
            # The CCCD layout always places the real name on the NEXT line(s),
            # not on the label line itself. When OCR lumps the label with
            # neighbouring text (e.g. "Hova teniFal nae" = "Họ và tên / Full
            # name" + corrupted continuation), the remainder after stripping
            # label tokens is garbage like "IFAL NAE" that accidentally
            # passes is_name_line. Don't use remainder — always prefer the
            # next 1–3 lines. On a clean scan the remainder is blank anyway.
            for j in range(i + 1, min(i + 4, len(lines))):
                if is_name_line(lines[j]):
                    full_name = clean_upper_vn_name(lines[j])
                    break
            # Edge case fallback: the remainder is only accepted if it's
            # decisively a well-formed name (≥3 words, ≥10 letters, and
            # none of the tokens are known label words). This handles the
            # rare well-OCR'd case where "Họ và tên: NGUYỄN VĂN A" ends up
            # on one line.
            if not full_name:
                remainder = line
                remainder = re.sub(r"(?i)ho\s*va\s*ten", "", remainder)
                remainder = re.sub(r"(?i)full\s*name", "", remainder)
                remainder = clean_upper_vn_name(remainder)
                rem_words = [w for w in remainder.split() if w]
                rem_letters = sum(len(w) for w in rem_words)
                if (len(rem_words) >= 3
                        and rem_letters >= 10
                        and not _is_label_line(remainder)
                        and is_name_line(remainder)):
                    full_name = remainder
            if full_name:
                break

    if not full_name and identity_number:
        # Locate the line containing the ID, then scan a window AFTER it for
        # the best name candidate. We no longer pick the FIRST match, because
        # poor-quality OCR sprinkles 2-char fragments like "G OH" between the
        # ID number and the real name. Pick the candidate with the highest
        # name_quality_score (more letters + more words + VN diacritics).
        id_idx = -1
        for i, line in enumerate(lines):
            if identity_number in re.sub(r"[^\d]", "", line):
                id_idx = i
                break
        if id_idx >= 0:
            best_line = None
            best_score = 0
            # Stop early if we hit a DOB-like pattern (name always precedes DOB on front).
            dob_re = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
            for line in lines[id_idx + 1: id_idx + 8]:
                if dob_re.search(line):
                    break
                score = name_quality_score(line)
                if score > best_score:
                    best_score = score
                    best_line = line
            if best_line is not None:
                full_name = clean_upper_vn_name(best_line)

    if not full_name:
        full_name = parse_name_from_mrz(lines)

    date_of_birth = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "ngay sinh" in norm or "date of birth" in norm:
            date_of_birth = extract_date_any(line)
            if not date_of_birth:
                for j in range(i + 1, min(i + 3, len(lines))):
                    date_of_birth = extract_date_any(lines[j])
                    if date_of_birth:
                        break
            if date_of_birth:
                break

    if not date_of_birth:
        dates = extract_all_dates(text)
        if dates:
            date_of_birth = dates[0]

    if not date_of_birth:
        date_of_birth = extract_dob_from_mrz(lines)

    gender = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "gioi tinh" in norm or "sex" in norm:
            if re.search(r"\bnam\b|\bmale\b", norm, re.I):
                gender = "Nam"
            elif re.search(r"\bnu\b|\bfemale\b", norm, re.I):
                gender = "Nữ"
            else:
                for j in range(i + 1, min(i + 3, len(lines))):
                    n2 = normalize_for_matching(lines[j])
                    if re.search(r"\bnam\b|\bmale\b", n2, re.I):
                        gender = "Nam"
                        break
                    if re.search(r"\bnu\b|\bfemale\b", n2, re.I):
                        gender = "Nữ"
                        break
            if gender:
                break

    if not gender:
        for line in lines:
            norm = normalize_for_matching(line)
            tokens = norm.split()
            for idx, token in enumerate(tokens):
                if token == "nam":
                    if idx > 0 and tokens[idx - 1] == "viet":
                        continue
                    gender = "Nam"
                    break
                if token == "nu":
                    gender = "Nữ"
                    break
            if gender:
                break

    if not gender:
        gender = extract_gender_from_mrz(lines)

    place_of_origin = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "que quan" in norm or "place of origin" in norm:
            remainder = line
            remainder = re.sub(r"(?i)qu[eê]\s*qu[aá]n", "", remainder)
            remainder = re.sub(r"(?i)place\s*of\s*origin", "", remainder)
            remainder = clean_field_value(remainder)
            if remainder:
                place_of_origin = remainder
            else:
                chunks = []
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = clean_field_value(lines[j])
                    if val:
                        chunks.append(val)
                if chunks:
                    place_of_origin = ", ".join(chunks)
            break

    address = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "noi thuong tru" in norm or "place of residence" in norm or "place of resid" in norm:
            remainder = line
            remainder = re.sub(r"(?i)n[oơ]i\s*th[uư][oờ]ng\s*tr[uú]", "", remainder)
            remainder = re.sub(r"(?i)place\s*of\s*residence", "", remainder)
            remainder = re.sub(r"(?i)place\s*of\s*resid\w*", "", remainder)
            remainder = clean_field_value(remainder)

            chunks = []
            if remainder:
                chunks.append(remainder)

            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = clean_field_value(lines[j])
                nxt_norm = normalize_for_matching(nxt)
                if not nxt:
                    continue
                if ("ngay cap" in nxt_norm or "place of issue" in nxt_norm or
                        "dac diem" in nxt_norm or "que quan" in nxt_norm):
                    break
                chunks.append(nxt)

            if chunks:
                address = ", ".join(chunks)
            break

    front_score, back_score = compute_side_scores(text)
    is_front_side = front_score >= back_score and (identity_number is not None or full_name is not None)

    # De-glue names returned as a single token. OCR drops spaces in the
    # name row on ~30% of phone uploads (tight card layouts). Splitting
    # at known VN syllable boundaries restores the displayable form.
    # Safe no-op for already-spaced names.
    if full_name:
        full_name = deglue_vn_name(full_name)

    logger.info(
        "[FRONT parse] isFront=%s frontScore=%d backScore=%d | id=%s name=%s dob=%s gender=%s origin=%s address=%s",
        is_front_side, front_score, back_score,
        identity_number or "MISSING",
        full_name or "MISSING",
        date_of_birth or "MISSING",
        gender or "MISSING",
        place_of_origin or "MISSING",
        address or "MISSING",
    )

    return {
        "side": "front",
        "identityNumber": identity_number,
        "fullName": full_name,
        "dateOfBirth": date_of_birth,
        "gender": gender,
        "nationality": "Việt Nam",
        "placeOfOrigin": place_of_origin,
        "address": address,
        "isFrontSide": is_front_side,
        "frontScore": front_score,
        "backScore": back_score,
        "rawText": text,
        "lines": lines,
    }


def parse_cccd_back(lines: List[str]) -> Dict:
    text = compact_join(lines)

    # Ngày cấp
    issue_date = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "ngay cap" in norm or "date of issue" in norm:
            issue_date = extract_date_any(line)
            if not issue_date:
                for j in range(i + 1, min(i + 3, len(lines))):
                    issue_date = extract_date_any(lines[j])
                    if issue_date:
                        break
            if issue_date:
                break

    if not issue_date:
        dates = extract_all_dates(text)
        if dates:
            issue_date = dates[-1]

    issue_place = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "noi cap" in norm or "place of issue" in norm:
            remainder = line
            remainder = re.sub(r"(?i)n[oơ]i\s*c[aấ]p", "", remainder)
            remainder = re.sub(r"(?i)place\s*of\s*issue", "", remainder)
            remainder = clean_field_value(remainder)
            if remainder:
                issue_place = remainder
            else:
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = clean_field_value(lines[j])
                    if val:
                        issue_place = val
                        break
            break

    if not issue_place:
        candidate_lines = []
        for line in lines:
            if looks_like_common_issue_place(line):
                candidate_lines.append(clean_field_value(line))

        if candidate_lines:
            issue_place = " ".join(candidate_lines)
            issue_place = re.sub(r"\s+", " ", issue_place).strip()

    if not issue_place and re.search(r"IDVNM|VNM<<", text, re.I):
        issue_place = COMMON_ISSUE_PLACE

    identifying_features = None
    for i, line in enumerate(lines):
        norm = normalize_for_matching(line)
        if "dac diem" in norm or "identifying feature" in norm or "identifying features" in norm:
            remainder = line
            remainder = re.sub(r"(?i)d[aặ]c\s*di[eể]m", "", remainder)
            remainder = re.sub(r"(?i)identifying\s*features?", "", remainder)
            remainder = clean_field_value(remainder)
            if remainder:
                identifying_features = remainder
            else:
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = clean_field_value(lines[j])
                    if val:
                        identifying_features = val
                        break
            break

    front_score, back_score = compute_side_scores(text)
    has_cccd_number = bool(extract_identity_number(text))
    is_readable = len(lines) >= 3 and len(text.strip()) >= 20
    negative_ok = (not has_cccd_number) and (issue_date is not None) and is_readable
    is_back_side = (back_score >= front_score or negative_ok) and is_readable

    logger.info(
        "[BACK parse] isBack=%s readable=%s frontScore=%d backScore=%d | issueDate=%s issuePlace=%s features=%s",
        is_back_side, is_readable, front_score, back_score,
        issue_date or "MISSING",
        issue_place or "MISSING",
        identifying_features or "MISSING",
    )

    return {
        "side": "back",
        "issueDate": issue_date,
        "issuePlace": issue_place,
        "identifyingFeatures": identifying_features,
        "isBackSide": is_back_side,
        "isReadable": is_readable,
        "frontScore": front_score,
        "backScore": back_score,
        "rawText": text,
        "lines": lines,
    }


def parse_auto(lines: List[str]) -> Dict:
    text = compact_join(lines)
    front_score, back_score = compute_side_scores(text)

    has_12_digits = bool(extract_identity_number(text))
    has_mrz = bool(re.search(r"IDVNM|VNM<<", text, re.I))

    if has_12_digits or front_score >= back_score:
        front = parse_cccd_front(lines)
        front["detectedSide"] = "front"
        return front

    if has_mrz or back_score > front_score:
        back = parse_cccd_back(lines)
        back["detectedSide"] = "back"
        return back

    # fallback
    front = parse_cccd_front(lines)
    if front.get("identityNumber") or front.get("fullName"):
        front["detectedSide"] = "front"
        return front

    back = parse_cccd_back(lines)
    back["detectedSide"] = "back"
    return back


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/cccd/quick-check")
async def ocr_cccd_quick_check(image: UploadFile = File(...)):
    """Lightweight CCCD verifier — returns ONLY the 3 fields the contract
    flow needs (ID, full name, expiry date) plus an `isCCCD` gate so the
    caller can short-circuit on non-CCCD uploads.

    Crops the image to the upper ~70% (header + ID block + name + dates) and
    downscales to a 480 longest-edge so det+rec runs on roughly a third of
    the pixels vs /ocr/cccd/front. Targets <10s on the 28C CPU box, vs
    ~25s for the full-detail endpoint.

    Skip rationale: address/place-of-origin/gender/DoB/MRZ are deliberately
    not parsed here. Use /ocr/cccd/front when the contract creation flow
    needs them."""
    try:
        raw = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Read upload failed: {e}")

    # Pre-crop + downscale BEFORE handing off to the OCR worker. Doing this
    # in the FastAPI process is fine — PIL is fast and the worker only
    # decodes/resizes once instead of twice.
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        # Header ~5% from the top is the SRV/Independence motto block — pure
        # noise for our 3 fields. Bottom 25-30% is the residence address +
        # photo border + expiry date. Cut the band [5%, 80%] which keeps
        # ID, name, DoB AND the "Có giá trị đến" (expiry) line on the
        # 2021+ CCCD layout.
        top_band = img.crop((0, int(h * 0.05), w, int(h * 0.80)))
        if max(top_band.size) > 480:
            top_band.thumbnail((480, 480), Image.LANCZOS)
        buf = io.BytesIO()
        top_band.save(buf, format="JPEG", quality=92)
        cropped_bytes = buf.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocess failed: {e}")

    loop = asyncio.get_running_loop()
    try:
        async with _verify_lock:
            global _ocr_executor
            if _ocr_executor is None:
                raise HTTPException(status_code=503, detail="OCR pool not ready")
            t0 = _time.perf_counter()
            # The worker also runs its own preprocess_edge resize; pass 480 so
            # we don't accidentally upscale our already-cropped band.
            lines = await loop.run_in_executor(
                _ocr_executor, _worker_ocr_lines, cropped_bytes, 480)
            elapsed_ms = round((_time.perf_counter() - t0) * 1000, 1)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    text = compact_join(lines)
    norm = normalize_for_matching(text)

    # Gate: at least 4 distinct CCCD-front keywords must appear. Random
    # selfies / passports / handwritten notes typically score 0-2.
    front_score = sum(1 for kw in FRONT_KEYWORDS if re.search(kw, norm, re.I))
    is_cccd = front_score >= 4

    if not is_cccd:
        logger.info(
            "[QUICK_CHECK] REJECT not-CCCD frontScore=%d elapsed_ms=%.1f",
            front_score, elapsed_ms,
        )
        return {
            "isCCCD": False,
            "frontScore": front_score,
            "identityNumber": None,
            "fullName": None,
            "expiryDate": None,
            "elapsedMs": elapsed_ms,
        }

    identity_number = extract_identity_number(text)

    # Name: scan AFTER the "Họ và tên / Full name" label line for the first
    # name-shaped line; fall back to the MRZ block on the back face if the
    # front got cropped off and the worker happened to still see MRZ.
    full_name = None
    for i, line in enumerate(lines):
        n = normalize_for_matching(line)
        if re.search(r"ho\s*va\s*ten|full\s*name", n, re.I):
            for j in range(i + 1, min(i + 4, len(lines))):
                if is_name_line(lines[j]):
                    full_name = clean_upper_vn_name(lines[j])
                    break
            if full_name:
                break
    if not full_name:
        full_name = parse_name_from_mrz(lines)

    # Expiry: prefer a line carrying "có giá trị đến" / "date of expiry"
    # (OCR often mangles to "cogla tj den" / "date ofoxpiry"). Fall back to
    # the latest date in the cropped region — DoB is in the past, expiry
    # is in the future, so the max date is the safest heuristic.
    expiry_date = None
    expiry_label_re = re.compile(
        r"(c[oa0]\s*g[li1]a?\s*tr[ji1]?\s*den)"   # "co gia tri den" variants
        r"|(date\s*o[fr]?\s*expir)"                  # English label
        r"|(\bden\s*\d{1,2}/)",                     # bare "den DD/..."
        re.I,
    )
    for line in lines:
        if expiry_label_re.search(normalize_for_matching(line)):
            d = extract_date_any(line)
            if d:
                expiry_date = d
                break
    if not expiry_date:
        all_dates = extract_all_dates(text)
        if all_dates:
            from datetime import datetime as _dt
            parsed = []
            for d in all_dates:
                try:
                    parsed.append((_dt.strptime(d, "%d/%m/%Y"), d))
                except ValueError:
                    continue
            if parsed:
                # Latest date wins — DoB is past, expiry is the maximum.
                parsed.sort(reverse=True)
                expiry_date = parsed[0][1]

    logger.info(
        "[QUICK_CHECK] OK frontScore=%d id=%s name=%s expiry=%s elapsed_ms=%.1f",
        front_score, identity_number or "MISSING",
        full_name or "MISSING", expiry_date or "MISSING", elapsed_ms,
    )

    return {
        "isCCCD": True,
        "frontScore": front_score,
        "identityNumber": identity_number,
        "fullName": full_name,
        "expiryDate": expiry_date,
        "elapsedMs": elapsed_ms,
    }


@app.post("/ocr/cccd/front")
async def ocr_cccd_front(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        lines = run_best_ocr(contents)
        return parse_cccd_front(lines)
    except Exception as e:
        logger.exception("[FRONT] OCR failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/cccd/back")
async def ocr_cccd_back(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        lines = run_best_ocr(contents)
        return parse_cccd_back(lines)
    except Exception as e:
        logger.exception("[BACK] OCR failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/cccd")
async def ocr_cccd_auto(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        lines = run_best_ocr(contents)
        return parse_auto(lines)
    except Exception as e:
        logger.exception("[AUTO] OCR failed")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Passport OCR — ICAO Doc 9303 TD3 machine-readable zone (two 44-char lines)
# =============================================================================

# ISO 3166-1 alpha-3 -> display name. Small map covering the countries our
# tenant base actually uses; extend as needed.
COUNTRY_NAMES = {
    "VNM": "Việt Nam",
    "USA": "United States",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "KOR": "Korea (Republic of)",
    "CHN": "China",
    "TWN": "Taiwan",
    "HKG": "Hong Kong",
    "SGP": "Singapore",
    "MYS": "Malaysia",
    "THA": "Thailand",
    "IDN": "Indonesia",
    "PHL": "Philippines",
    "IND": "India",
    "AUS": "Australia",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Germany",
    "RUS": "Russia",
    "NLD": "Netherlands",
    "ITA": "Italy",
    "ESP": "Spain",
    "NZL": "New Zealand",
}


def _mrz_clean(s: str) -> str:
    """Keep only the alphabet the MRZ uses (A–Z, 0–9, <)."""
    return re.sub(r"[^A-Z0-9<]", "", s.upper())


def _pick_mrz_td3(lines: List[str]) -> Optional[Tuple[str, str]]:
    """Return (line1, line2) if we can find a TD3 MRZ pair among OCR output."""
    candidates = [_mrz_clean(l) for l in lines]
    candidates = [c for c in candidates if len(c) >= 30]

    # Line 1 of a TD3 MRZ starts with P (passport) and has surname<<given
    l1_idx = -1
    for i, c in enumerate(candidates):
        if c.startswith("P") and "<<" in c:
            l1_idx = i
            break
    if l1_idx == -1 or l1_idx + 1 >= len(candidates):
        return None

    l1 = candidates[l1_idx][:44].ljust(44, "<")
    l2 = candidates[l1_idx + 1][:44].ljust(44, "<")
    return l1, l2


def _parse_mrz_date(yymmdd: str) -> Optional[str]:
    """YYMMDD -> DD/MM/YYYY (assume 20xx when YY < 50, else 19xx)."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy = int(yymmdd[:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]
    # Expiry / issue dates after 2050 are basically impossible for current
    # passports; DOB before 1950 is rare but can happen. 50 is the accepted
    # ICAO rule-of-thumb cutoff.
    century = 2000 if yy < 50 else 1900
    year = century + yy
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31):
        return None
    return f"{dd}/{mm}/{year}"


def parse_passport_mrz(lines: List[str]) -> Dict:
    """
    Parse TD3 MRZ. Fields per ICAO 9303-4:
      Line 1: P<ISSUE_COUNTRY SURNAME<<GIVEN_NAMES
      Line 2: PASSPORT_NO[9] CHK NATIONALITY[3] DOB[6] SEX[1] EXPIRY[6] PERSONAL[14] CHK CHK
    """
    pair = _pick_mrz_td3(lines)
    if pair is None:
        return {
            "passportNumber": None,
            "fullName": None,
            "surname": None,
            "givenName": None,
            "dateOfBirth": None,
            "gender": None,
            "nationality": None,
            "countryCode": None,
            "issueDate": None,
            "issuePlace": None,
            "expiryDate": None,
            "mrz": None,
            "rawLines": lines,
        }

    l1, l2 = pair
    mrz_joined = l1 + "\n" + l2

    # --- Line 1 ---
    # P<ISSUE_COUNTRY(3)NAMES...
    issue_country = l1[2:5].replace("<", "")
    names_field = l1[5:].replace("<", " ").strip()
    surname, _, given = names_field.partition("  ")  # double-< => 2 spaces
    if not given:
        # Some OCR merges the double-< into single space
        parts = names_field.split(" ", 1)
        surname = parts[0]
        given = parts[1] if len(parts) > 1 else ""
    surname = re.sub(r"\s+", " ", surname).strip()
    given = re.sub(r"\s+", " ", given).strip()
    full_name = (surname + " " + given).strip()

    # --- Line 2 ---
    passport_no = l2[0:9].replace("<", "").strip() or None
    nationality_code = l2[10:13].replace("<", "")
    dob_raw = l2[13:19]
    sex_char = l2[20:21]
    expiry_raw = l2[21:27]

    date_of_birth = _parse_mrz_date(dob_raw)
    expiry_date = _parse_mrz_date(expiry_raw)

    gender = None
    if sex_char == "M":
        gender = "Nam"
    elif sex_char == "F":
        gender = "Nữ"

    nationality_display = COUNTRY_NAMES.get(nationality_code, nationality_code or None)
    country_code = issue_country or nationality_code or None

    # Issue date / place are in the visual zone only. Best-effort scan.
    visible_text = "\n".join(lines)
    issue_date = None
    for kw in [r"date\s*of\s*issue", r"ngay\s*cap", r"issue\s*date"]:
        if re.search(kw, normalize_for_matching(visible_text), re.I):
            for line in lines:
                if re.search(kw, normalize_for_matching(line), re.I):
                    issue_date = extract_date_any(line)
                    if not issue_date:
                        idx = lines.index(line)
                        for j in range(idx + 1, min(idx + 3, len(lines))):
                            issue_date = extract_date_any(lines[j])
                            if issue_date:
                                break
                    if issue_date:
                        break
            if issue_date:
                break

    issue_place = None
    for kw in [r"authority", r"place\s*of\s*issue", r"noi\s*cap"]:
        for line in lines:
            if re.search(kw, normalize_for_matching(line), re.I):
                remainder = re.sub(kw, "", line, flags=re.I)
                remainder = clean_field_value(remainder)
                if remainder and not re.search(r"^\d", remainder):
                    issue_place = remainder
                    break
        if issue_place:
            break

    logger.info(
        "[PASSPORT parse] no=%s name=%s nat=%s dob=%s gender=%s expiry=%s issuer=%s",
        passport_no, full_name, nationality_display,
        date_of_birth, gender, expiry_date, country_code,
    )

    return {
        "passportNumber": passport_no,
        "fullName": full_name or None,
        "surname": surname or None,
        "givenName": given or None,
        "dateOfBirth": date_of_birth,
        "gender": gender,
        "nationality": nationality_display,
        "countryCode": country_code,
        "issueDate": issue_date,
        "issuePlace": issue_place,
        "expiryDate": expiry_date,
        "mrz": mrz_joined,
        "rawLines": lines,
    }


@app.post("/ocr/passport")
async def ocr_passport(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        lines = run_best_ocr(contents)
        return parse_passport_mrz(lines)
    except Exception as e:
        logger.exception("[PASSPORT] OCR failed")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# /ocr/cccd/verify — quick KYC: front must be a CCCD front and expose ID/name;
# back only needs to look like a CCCD back. It does not require back-side MRZ
# ID/name matching because the EContract service compares the front ID/name
# against the signed contract. Front + back OCR still runs in parallel.
# =============================================================================

def _ocr_lines_with_instance(instance: PaddleOCR, img: np.ndarray) -> List[str]:
    results = instance.predict(img)
    out: List[str] = []
    if not results:
        return out
    for res in results:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        for text, score in zip(texts, scores):
            norm = normalize_text_basic(text).strip()
            if norm:
                out.append(norm)
    return out


def _normalize_name_key(name: Optional[str]) -> str:
    """Comparison key: strip diacritics, uppercase, keep only A-Z. Handles
    the common case where OCR glues 'DUC HIEU' → 'DUCHIEU' on the front
    while MRZ gives 'DUC HIEU' spaced — both collapse to 'DUCHIEU'."""
    if not name:
        return ""
    s = strip_accents(name).upper()
    return re.sub(r"[^A-Z]", "", s)


# Label tokens that show up when parse_cccd_front picked a label row as the
# name. Seen in the wild on real CCCD uploads:
#   "NGAY SINHI DATE DF SIR"  (Ngày sinh / Date of birth label row)
#   "HO VA TEN FULL NAME"
#   "QUE QUAN PLACE OF ORIGIN"
# When the front name contains any of these after diacritic strip, we treat
# it as unreadable and fall back to the back-side MRZ name for matching.
_FRONT_NAME_LABEL_TOKENS = {
    "NGAY", "SINH", "DATE", "BIRTH", "SIR",
    "HOVA", "TEN", "FULL", "NAME",
    "QUE", "QUAN", "PLACE", "ORIGIN",
    "NOI", "THUONG", "TRU", "RESIDENCE",
    "GIOI", "TINH", "SEX", "QUOC", "NATIONALITY",
    "CITIZEN", "IDENTITY", "CARD",
}


def _front_name_looks_like_label(name: Optional[str]) -> bool:
    """True when the OCR'd front name is actually a field label the parser
    mistook for a name. Compare against `_FRONT_NAME_LABEL_TOKENS` on the
    upper-case accent-stripped tokens; any hit means the value is garbage."""
    if not name:
        return True
    tokens = strip_accents(name).upper().split()
    return any(tok in _FRONT_NAME_LABEL_TOKENS for tok in tokens)


def _parse_verify_side(lines: List[str], expected_side: str, elapsed_ms: float) -> Dict:
    """Parse OCR text lines into the /verify response envelope. Pulled out
    of _process_verify_side so the heavy OCR runs in a child process while
    parsing stays in the main process."""

    text = compact_join(lines)
    front_score, back_score = compute_side_scores(text)
    has_mrz = bool(re.search(r"IDVNM|VNM<<", text, re.I))

    # Non-CCCD guard: no keywords, no MRZ, no 12-digit id → not a card at all.
    if front_score == 0 and back_score == 0 and not has_mrz and not extract_identity_number(text):
        logger.info("verify side=%s NOT_CCCD lines=%d elapsed_ms=%.1f",
                    expected_side, len(lines), elapsed_ms)
        return {
            "sideOk": False,
            "detectedSide": "unknown",
            "identityNumber": None,
            "fullName": None,
            "reason": "NOT_CCCD",
            "frontScore": 0,
            "backScore": 0,
            "elapsedMs": elapsed_ms,
        }

    if expected_side == "front":
        parsed = parse_cccd_front(lines)
        is_front = bool(parsed.get("isFrontSide"))
        detected = "front" if is_front else ("back" if parsed.get("backScore", 0) > parsed.get("frontScore", 0) else "unknown")
        return {
            "sideOk": is_front,
            "detectedSide": detected,
            "identityNumber": parsed.get("identityNumber"),
            "fullName": parsed.get("fullName"),
            "dateOfBirth": parsed.get("dateOfBirth"),
            "frontScore": parsed.get("frontScore"),
            "backScore": parsed.get("backScore"),
            "reason": None if is_front else "WRONG_SIDE",
            "elapsedMs": elapsed_ms,
        }

    # Back side quick check: parse enough text to know this is probably a CCCD
    # back. Do not require issue date, MRZ ID, or MRZ name; those checks made
    # tenant confirmation slow and brittle while adding little value here.
    parsed = parse_cccd_back(lines)
    mrz_id = extract_identity_number(text)
    mrz_name = parse_name_from_mrz(lines)
    is_readable = len(lines) >= 2 and len(text.strip()) >= 12
    front_score = parsed.get("frontScore", 0)
    back_score = parsed.get("backScore", 0)
    has_date_like_text = bool(re.search(r"\d{2}/\d{2}/\d{4}", normalize_date_text(text)))
    # Date text alone is weak because the front side has date of birth/expiry.
    # Allow it only when the image does not look more like the front side.
    is_back = bool(parsed.get("isBackSide")) or bool(
        is_readable and (
            has_mrz
            or back_score > 0
            or (has_date_like_text and back_score >= front_score)
        )
    )
    detected = "back" if is_back else ("front" if front_score > back_score else "unknown")
    return {
        "sideOk": is_back,
        "detectedSide": detected,
        "identityNumber": mrz_id,
        "fullName": mrz_name,
        "issueDate": parsed.get("issueDate"),
        "issuePlace": parsed.get("issuePlace"),
        "frontScore": parsed.get("frontScore"),
        "backScore": parsed.get("backScore"),
        "reason": None if is_back else "WRONG_SIDE",
        "elapsedMs": elapsed_ms,
    }


@app.post("/ocr/cccd/verify")
async def ocr_cccd_verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
):
    """Quick CCCD check: `front` must be front-side CCCD with ID + name,
    `back` must roughly look like back-side CCCD. The contract service does
    the authoritative front ID/name comparison against contract data."""
    try:
        front_bytes = await front.read()
        back_bytes = await back.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Read upload failed: {e}")

    t0 = _time.perf_counter()
    # True parallel via ProcessPoolExecutor — each worker has its own
    # Paddle/OpenMP pool, so front and back OCR actually overlap on CPU
    # instead of fighting a shared thread pool. Lock serialises overlapping
    # /verify calls so a second caller doesn't bypass the 2-worker budget.
    loop = asyncio.get_running_loop()
    try:
        async with _verify_lock:
            global _ocr_executor
            if _ocr_executor is None:
                raise HTTPException(status_code=503, detail="OCR pool not ready")
            front_task = loop.run_in_executor(
                _ocr_executor, _worker_ocr_lines, front_bytes, _VERIFY_PREPROCESS_EDGE)
            back_task = loop.run_in_executor(
                _ocr_executor, _worker_ocr_lines, back_bytes, _VERIFY_PREPROCESS_EDGE)
            t_front_start = _time.perf_counter()
            front_lines = await front_task
            front_elapsed_ms = round((_time.perf_counter() - t_front_start) * 1000, 1)
            t_back_start = _time.perf_counter()
            back_lines = await back_task
            back_elapsed_ms = round((_time.perf_counter() - t_back_start) * 1000, 1)
            front_res = _parse_verify_side(front_lines, "front", front_elapsed_ms)
            back_res = _parse_verify_side(back_lines, "back", back_elapsed_ms)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[VERIFY] OCR failed")
        raise HTTPException(status_code=500, detail=str(e))

    f_id = front_res.get("identityNumber")
    f_name = front_res.get("fullName")
    id_match = bool(f_id)
    name_match = bool(f_name) and not _front_name_looks_like_label(f_name)

    passed = bool(front_res.get("sideOk") and back_res.get("sideOk") and id_match and name_match)

    reasons: List[str] = []
    if front_res.get("reason"):
        reasons.append(f"front:{front_res['reason']}")
    if back_res.get("reason"):
        reasons.append(f"back:{back_res['reason']}")
    if front_res.get("sideOk") and back_res.get("sideOk"):
        if not id_match:
            reasons.append("ID_MISSING")
        if not name_match:
            reasons.append("NAME_MISSING")

    total_ms = round((_time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "[VERIFY] pass=%s id=%s frontNameOk=%s reasons=%s total_ms=%.1f",
        passed, f_id or "MISSING", name_match,
        ",".join(reasons) or "-", total_ms,
    )

    return {
        "pass": passed,
        "reasons": reasons,
        "match": {
            "idMatch": id_match,
            "nameMatch": name_match,
            "identityNumber": f_id,
            "fullName": f_name,
        },
        "front": front_res,
        "back": back_res,
        "totalMs": total_ms,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
