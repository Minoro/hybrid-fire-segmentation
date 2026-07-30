ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libjpeg-dev libpng-dev \
    software-properties-common \
    build-essential \
    wget \
    && add-apt-repository -y ppa:ubuntugis/ppa \
    && apt-get update \
    && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /hybrid-fire-segmentation/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

RUN pip install --no-cache-dir torchmetrics einops torchsummary jupyterlab torchvision torchgeo terratorch

RUN python3 -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))"


CMD [ "/bin/bash" ]