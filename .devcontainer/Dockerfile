# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install --no-install-recommends python3.10 python3-pip git

# Install pip packages
RUN mkdir /init
COPY ./.devcontainer/requirements.txt /init/requirements.txt
RUN pip -q install pip --upgrade
RUN pip install -r /init/requirements.txt



