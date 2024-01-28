FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Prevent prompts for user input during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3.12 \
        python3-pip

# Set the working directory
WORKDIR /app

COPY . /app

# Install required python packages

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

# Set the entrypoint
ENTRYPOINT [ "python3", "app.py" ]
