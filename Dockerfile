FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /home/transformer

RUN pip install --upgrade pip
RUN pip install keras-transformer
