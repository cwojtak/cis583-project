FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Prevent prompts for user input during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.12 \
        python3-pip

# Install required python packages
COPY requirements.txt /app

RUN python3 -m pip install -r requirements.txt

# Copy project into container
COPY . /app

# Set the entrypoint
ENTRYPOINT [ "python3", "app.py" ]
