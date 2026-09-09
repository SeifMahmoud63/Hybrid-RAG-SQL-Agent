# Use official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements file first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY ./ /code/

# Command to run the FastAPI server via uvicorn
CMD ["uvicorn", "routes.data:app", "--host", "0.0.0.0", "--port", "80"]