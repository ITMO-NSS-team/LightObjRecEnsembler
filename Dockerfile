FROM nvcr.io/nvidia/pytorch:21.06-py3
RUN pip install torchvision matplotlib ujson ipywidgets pytorch-lightning

RUN mkdir -p /workspace/NAS-object-recognition
COPY ./NAS-object-recognition /workspace/NAS-object-recognition