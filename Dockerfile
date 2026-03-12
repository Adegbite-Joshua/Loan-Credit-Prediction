FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir gunicorn==21.2.0


COPY . .

# Expose port
EXPOSE 10000

# Run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT