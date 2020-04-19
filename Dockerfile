FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY launcher launcher
RUN pip install file:./launcher
RUN rm -rf launcher

VOLUME /src
VOLUME /data
VOLUME /output
WORKDIR /src