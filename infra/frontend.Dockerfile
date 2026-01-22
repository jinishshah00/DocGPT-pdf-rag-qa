FROM python:3.11-slim
WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir --prefer-binary --default-timeout=120 --retries 5 -r /app/frontend/requirements.txt

COPY frontend /app/frontend
COPY backend /app/backend
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "/app/frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
