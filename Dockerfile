FROM python:3.10-slim

# 1. منع بايثون من الـ buffering وتوليد كاش pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 2. تسطيب مكتبات النظام الضرورية لـ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. إنشاء مستخدم non-root
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# 4. تثبيت مكتبات بايثون
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. نسخ باقي ملفات المشروع ونقل الملكية للـ appuser
COPY . .
RUN chown -R appuser:appgroup /app

# 6. التبديل إلى المستخدم العادي
USER appuser

EXPOSE 7860
EXPOSE 8000

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]