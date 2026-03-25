from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import re
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu"
)


def parse_cccd(lines: list[str]) -> dict:
    text = "\n".join(lines)

    def extract(pattern):
        m = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        return m.group(1).strip() if m else None

    return {
        "identityNumber": extract(r'(?:S[oố]|No|ID)[\s:\.]*([0-9]{9,12})'),
        "fullName":       extract(r'(?:H[oọ] v[aà] t[eê]n|Full name)[\s:]*([^\n]+)'),
        "dateOfBirth":    extract(r'(?:Ng[aà]y sinh|Date of birth)[\s:]*(\d{2}/\d{2}/\d{4})'),
        "gender":         extract(r'(?:Gi[oớ]i t[ií]nh|Sex)[\s:]*([^\n/]+?)(?:\s*/|\s*Qu[oô]c|\n)'),
        "address":        extract(r'(?:N[oơ]i th[uư][oờ]ng tr[uú]|Place of residence)[\s:]*([^\n]+)'),
        "issueDate":      extract(r'(?:Ng[aà]y c[aấ]p|Date of issue)[\s:]*(\d{2}/\d{2}/\d{4})'),
        "issuePlace":     extract(r'(?:N[oơ]i c[aấ]p|Place of issue)[\s:]*([^\n]+)'),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/cccd")
async def ocr_cccd(image: UploadFile = File(...)):
    try:
        contents = await image.read()

        results = ocr.predict(io.BytesIO(contents))

        lines = []
        for res in results:
            for item in res.get("rec_texts", []):
                if item:
                    lines.append(item)

        logger.info(f"OCR extracted {len(lines)} lines")
        parsed = parse_cccd(lines)
        logger.info(f"Parsed identity: {parsed.get('identityNumber')}, name: {parsed.get('fullName')}")

        return parsed
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)