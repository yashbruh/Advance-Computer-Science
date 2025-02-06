# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container

COPY ./requirements.txt ./requirements.txt
COPY ./soy-analog-447121-p3-6f72d899b2c6.json ./soy-analog-447121-p3-6f72d899b2c6.json


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./static ./static
COPY ./templates ./templates
COPY ./app.py ./app.py
# Expose port 8080 (required for Cloud Run)
EXPOSE 8080

# Start the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
