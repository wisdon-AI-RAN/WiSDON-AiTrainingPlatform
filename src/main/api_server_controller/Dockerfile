FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=900 -r requirements.txt

# CMD ["uvicorn","main:app","--host","0.0.0.0","--port","2980"]
CMD ["python3","api_server_controller/runAPI.py"]