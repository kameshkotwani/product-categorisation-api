# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that the application will listen on.
EXPOSE 5000

# Run the FastAPI app with uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
