FROM python:3.9-slim

WORKDIR /app

# Absolute minimum dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .

# Install with minimal footprint
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy only essential files
COPY app.py ./
COPY model/precheck_model.h5 ./model/
COPY model/tb_classification_model.h5 ./model/
COPY model/tb_densenet_model.keras ./model/
COPY templates/ ./templates/

RUN mkdir -p static/uploads

# Nuclear cleanup
RUN find /usr/local -depth \
    \( -name '*.pyc' -o -name '*.pyo' -o -name '__pycache__' \) -exec rm -rf '{}' + \
    && rm -rf /root/.cache /tmp/*

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=1", "--timeout=120", "app:app"]