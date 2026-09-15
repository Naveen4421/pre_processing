FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_DEBUG=0

WORKDIR /app

# poppler-utils: PDF rasterization (pdftoppm/pdfinfo).
# tesseract-ocr: used ONLY for Orientation & Script Detection (--psm 0) to
# auto-correct 90/180/270 degree page rotation. No text recognition/OCR is
# performed by this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p webapp/uploads webapp/outputs

EXPOSE 5000

CMD ["python", "-m", "webapp.app"]
