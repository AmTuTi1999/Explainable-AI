# Use a Python base image
FROM python:3.9-slim

# Install pip and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy requirements file into the container
COPY requirements.txt .

# Expose port if needed (optional, e.g., for Flask or Django)
EXPOSE 8000

# Ensure Python and pip are up-to-date
RUN pip install --upgrade pip setuptools

# Install the Python dependencies
RUN pip install -r requirements.txt

# Switch to non-root user
ARG USERNAME=vscode
RUN useradd -m $USERNAME \
    && chown -R $USERNAME /workspace

# Switch to user vscode
USER vscode

# Set default command
CMD ["python3"]
