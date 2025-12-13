# Use python 3.11 slim base image
FROM python:3.11-slim
# set working directory
WORKDIR /app

#install system dep.
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application files
COPY app.py transformer.py ./

#copy model and tokenizer folders
COPY models ./models
COPY tokenizer ./tokenizer

#expose API port
EXPOSE 8000

#start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
