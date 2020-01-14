docker build . -t keras

docker run \
    -v $(pwd)/src:/src \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash keras

https://github.com/qubvel/segmentation_models
https://github.com/Diyago/ML-DL-scripts/tree/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline
