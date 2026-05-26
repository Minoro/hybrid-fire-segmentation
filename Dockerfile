ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libjpeg-dev libpng-dev\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /hybrid-fire-segmentation/


RUN conda install gdal -c conda-forge -y

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchmetrics einops torchsummary jupyterlab torchvision torchgeo terratorch

RUN python3 -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))"


CMD [ "/bin/bash" ]