from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import re
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)


def parse_cccd(lines: list[str]) -> dict:
    text = "\n".join(lines)

    def extract(pattern):
        m = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        return m.group(1).strip() if m else None

    return {
        "identityNumber": extract(r'(?:Số|So|ID|No)[\s:\.]*([0-9]{9,12})'),
        "fullName": extract(r'(?:Họ và tên|Ho va ten|Full name)[\s:]*([^\n]+)'),
        "dateOfBirth": extract(r'(?:Ngày sinh|Date of birth)[\s:]*(\d{2}/\d{2}/\d{4})'),
        "gender": extract(r'(?:Giới tính|Sex)[\s:]*([^\n/]+?)(?:\s*/|\s*Quốc|\n)'),
        "address": extract(r'(?:Nơi thường trú|Place of residence)[\s:]*([^\n]+)'),
        "issueDate": extract(r'(?:Ngày cấp|Date of expiry|Có giá trị đến)[\s:]*(\d{2}/\d{2}/\d{4})'),
        "issuePlace": extract(r'(?:Nơi cấp|Place of issue)[\s:]*([^\n]+)'),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/cccd")
async def ocr_cccd(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_image)

        result = ocr.ocr(img_array, cls=True)
        lines = [line[1][0] for block in result for line in block if line[1][1] > 0.5]

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
