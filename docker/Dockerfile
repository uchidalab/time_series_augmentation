FROM tensorflow/tensorflow:1.10.0-gpu-py3 
LABEL maintainer="Brian Kenji Iwana"

RUN apt-get update \
    && apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tqdm
RUN pip install keras==2.2.4
