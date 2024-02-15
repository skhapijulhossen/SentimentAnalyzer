# Use an official Python image with a specific version as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local code to the container at the working directory
COPY . /app

# Install any dependencies needed for your Streamlit app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip freeze 

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "run.py"]
