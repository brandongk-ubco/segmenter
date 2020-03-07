docker build . -t segmenter

# Build and split the dataset

# Load the Docker Container
docker run \
    -v $(pwd)/src:/src \
    -v $(pwd)/datasets/severstal/out:/data \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash segmenter
