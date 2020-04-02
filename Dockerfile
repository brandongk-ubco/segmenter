FROM tensorflow/tensorflow:nightly-gpu-py3

RUN pip install \
    albumentations \
    segmentation-models \
    psutil
