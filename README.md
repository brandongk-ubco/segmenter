docker build . -t keras

# Train the class segmentation models
docker run \
    -v $(pwd)/src:/src \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash keras

# Inference for the full dataset
docker run \
    -v $(pwd)/src:/src \
    -v /mnt/work/severstal/train:/data/images \
    -v /mnt/work/severstal/masks:/data/masks \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash keras

# Train the segmentation resolver
docker run \
    -v $(pwd)/src:/src \
    -v /mnt/work/severstal/masks:/data \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash keras

https://github.com/qubvel/segmentation_models
https://github.com/Diyago/ML-DL-scripts/tree/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline
