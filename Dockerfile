# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV (used by YOLO)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy your code and the YOLO model into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "VLMMain:app", "--host", "0.0.0.0", "--port", "8000"]