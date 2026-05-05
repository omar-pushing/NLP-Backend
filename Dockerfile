FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080
CMD ["sh", "start.sh"]
