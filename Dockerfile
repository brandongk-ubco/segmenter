FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install \
    scikit-image \
    scikit-learn \
    keras==2.3.1 \
    albumentations \
    segmentation-models

