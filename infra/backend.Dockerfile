FROM python:3.11-slim
WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
COPY backend/requirements.txt /app/backend/requirements.txt
# Upgrade pip tooling, then install dependencies with binary preference and sane network timeouts
RUN pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir --prefer-binary --default-timeout=120 --retries 5 -r /app/backend/requirements.txt
COPY backend /app/backend
ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]