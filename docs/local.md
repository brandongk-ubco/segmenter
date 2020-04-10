This setup assumes you are running Ubuntu 18.04.  Note that Docker works on Windows, and using WSL you can run Ubuntu commands pretty easily.  However, it's currently not possible to setup Docker to pass through Nvidia drivers, so it's not possible to use GPUs.

# Setting up Docker and NVidia Docker

## Install Docker

Docker can be installed in many ways.  On Ubuntu, the easiest is through `sudo snap install docker`.  However, if you plan on using GPUs, it's better to install the apt-get package `docker-ce`:

```
# From https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io
```

Either way, you then need to add yourself to the Docker group and logout/login:

```
# From https://docs.docker.com/install/linux/linux-postinstall/
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Install the NVidia Drivers

By default, Ubuntu uses nouveau drivers, which won't use your NVidia GPU.  To install the NVidia drivers:

```
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo ubuntu-drivers autoinstall
sudo apt-get -y install nvidia-cuda-toolkit gcc-6
sudo apt-get -y dist-upgrade
```

## Install the NVidia Container Toolkit

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) allows users to build and run GPU accelerated Docker containers.

```
# From https://github.com/NVIDIA/nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Install Docker

sudo snap install docker

## See if it all works

1. Restart the computer.
2. `docker run --gpus all nvidia/cuda nvidia-smi`

You should see all your NVidia GPUs listed.

# Install VSCode

`sudo snap install --classic code`

# Set up Miniconda (or Anaconda if you need a GUI)

```
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x miniconda.sh
./miniconda.sh -b
rm miniconda.sh
```

# Create the conda environment

`conda create -n segmenter python=3.8`

# Install the requirements

`conda activate segmenter && pip install -r requirements.txt`

# View the pydocs

`pydoc -b`

# Install Docker Dependencies

`docker build . -t segmenter`

# Load the Docker Container

```docker run \
    -v $(pwd)/src:/src \
    -v $(pwd)/datasets/severstal/out:/data \
    -v $(pwd)/output:/output \
    --gpus all \
    -u $(id -u):$(id -g) \
    -it --entrypoint bash segmenter```