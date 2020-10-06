FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

VOLUME /src
VOLUME /data
VOLUME /output
WORKDIR /src