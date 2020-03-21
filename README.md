# Install Anaconda or Miniconda

This is a platform-specific setup.

# Create a Conda environment

`conda create -n segmenter python=3.8`

# Install the requirements

`conda activate segmenter && pip install -r requirements.txt`

# View the pydocs

`pydoc -b`

# Install Docker Dependencies

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
