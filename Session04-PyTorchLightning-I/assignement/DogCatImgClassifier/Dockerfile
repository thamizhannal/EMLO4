#FROM mcr.microsoft.com/devcontainers/base:ubuntu
FROM python:3.12.0-slim
WORKDIR /app
COPY requirements.txt /app/
COPY data /app/
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . . 
CMD ["python", "src/train.py"]
