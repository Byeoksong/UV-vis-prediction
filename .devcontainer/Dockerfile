FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install common tools and dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    environment-modules \
    vim htop wget tcl gpg net-tools \
    openssh-client openssh-server \
    git libncurses5-dev libncursesw5-dev tmux bc \
    sudo # Keep sudo for potential future use, but remove NOPASSWD

# Create a non-root user (optional, but good practice)
# VS Code Dev Containers typically handle user creation, but if you need a specific user
# RUN useradd -m -s /bin/bash vscode && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode
# USER vscode

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /tmp/requirements.txt

# Set the default user to vscode (handled by devcontainer features)
# USER vscode

# Set the working directory
WORKDIR /workspaces/UV-vis_prediction
