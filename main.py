from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import numpy as np
import cv2
import re
import io
import logging
import unicodedata
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("cccd-ocr")

app = FastAPI(title="Vietnam CCCD OCR Service")

ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    use_gpu=False,
    show_log=False
)

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
    rgb = read_image_from_bytes(contents)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    rgb_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    rgb_otsu = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
    rgb_adapt = cv2.cvtColor(adapt, cv2.COLOR_GRAY2RGB)

    return [rgb, rgb_eq, rgb_otsu, rgb_adapt]


def ocr_lines_with_scores(img: np.ndarray) -> List[Tuple[str, float]]:
    results = ocr.ocr(img, cls=True)
    out = []
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2 and line[1]:
                text = normalize_text_basic(line[1][0])
                score = float(line[1][1]) if len(line[1]) > 1 else 0.0
                if text.strip():
                    out.append((text.strip(), score))
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
    candidates = preprocess_variants(contents)
    best_lines = []
    best_score = -1.0

    for idx, img in enumerate(candidates):
        lines_with_scores = ocr_lines_with_scores(img)
        score = score_ocr_quality(lines_with_scores)
        logger.info("OCR variant=%d lines=%d score=%.4f", idx, len(lines_with_scores), score)
        if score > best_score:
            best_score = score
            best_lines = [t for t, _ in lines_with_scores]

    logger.info("Chosen OCR lines (%d):\n%s",
                len(best_lines),
                "\n".join(f"  [{i}] {l}" for i, l in enumerate(best_lines)))
    return best_lines


def is_name_line(line: str) -> bool:
    stripped = clean_upper_vn_name(line)
    return (
            len(stripped) >= 4 and
            len(stripped.split()) >= 2 and
            not re.search(r"\d", stripped)
    )


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
    return None


def extract_mrz_lines(lines: List[str]) -> List[str]:
    mrz = []
    for line in lines:
        line2 = line.upper().replace(" ", "")
        if re.search(r"[A-Z0-9<]{15,}", line2):
            mrz.append(line2)
    return mrz[-3:] if len(mrz) >= 3 else mrz


def parse_name_from_mrz(lines: List[str]) -> Optional[str]:
    mrz_lines = extract_mrz_lines(lines)
    if not mrz_lines:
        return None
    for line in mrz_lines:
        if "<<" in line:
            parts = line.split("<<")
            if len(parts) >= 2:
                surname = parts[0].split("<")[-1]
                given = parts[1].replace("<", " ").strip()
                full = f"{surname} {given}".strip()
                full = re.sub(r"\s+", " ", full)
                if full:
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
            remainder = line
            remainder = re.sub(r"(?i)ho\s*va\s*ten", "", remainder)
            remainder = re.sub(r"(?i)full\s*name", "", remainder)
            remainder = clean_upper_vn_name(remainder)
            if is_name_line(remainder):
                full_name = remainder
                break
            for j in range(i + 1, min(i + 3, len(lines))):
                if is_name_line(lines[j]):
                    full_name = clean_upper_vn_name(lines[j])
                    break
            if full_name:
                break

    if not full_name and identity_number:
        id_idx = -1
        for i, line in enumerate(lines):
            if identity_number in re.sub(r"[^\d]", "", line):
                id_idx = i
                break
        if id_idx >= 0:
            for line in lines[id_idx + 1: id_idx + 7]:
                if is_name_line(line):
                    full_name = clean_upper_vn_name(line)
                    break

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
