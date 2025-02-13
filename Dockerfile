FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app
RUN mkdir -p /output

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD ["python", "./main.py"]
