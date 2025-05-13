FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    espeak \
    ffmpeg \
    git \
    libsndfile1 \
    && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 4401

CMD ["python", "manage.py", "runserver", "0.0.0.0:4401"]
