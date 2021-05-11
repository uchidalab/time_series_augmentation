FROM tensorflow/tensorflow:2.4.1-gpu-jupyter 
LABEL maintainer="Brian Kenji Iwana"

RUN apt-get update \
    && apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tqdm
RUN pip install keras==2.2.4
RUN pip install scikit-image==0.15.0
