# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables so Python doesn't buffer logs
ENV PYTHONUNBUFFERED=1

# Install Heavy System Dependencies required for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project directory into the container
COPY . .

# Expose port 8000 for the web server
EXPOSE 8000

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]