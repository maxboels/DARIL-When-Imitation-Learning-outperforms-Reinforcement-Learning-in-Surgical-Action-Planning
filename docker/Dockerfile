FROM nvcr.io/nvidia/pytorch:24.10-py3
ARG USER_ID
ARG GROUP_ID
ARG USER

ENV TZ=Europe/London

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && apt-get install -y sudo \
    libgl1-mesa-glx \
    ffmpeg \
    libimage-exiftool-perl \
    build-essential

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# Add user
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# Set working directory
WORKDIR /nfs/home/$USER

# Expose port for Jupyter Notebook
EXPOSE 8888

# Define environment variable
ENV PYTHONUNBUFFERED=1