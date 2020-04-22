This setup assumes you are running Ubuntu 18.04.  Note that Docker works on Windows, and using WSL you can run Ubuntu commands pretty easily.  However, it's currently not possible to setup Docker to pass through Nvidia drivers, so it's not possible to use GPUs.

# Install basics

```
sudo apt-get -y update
sudo apt-get -y install build-essential git
```

# Set up Miniconda (or Anaconda if you need a GUI)

```
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x miniconda.sh
./miniconda.sh -b
rm miniconda.sh
~/miniconda3/bin/conda init
```

# Create the conda environment

`conda create -n segmenter python=3.6`

# Install the requirements

`conda activate segmenter && pip install -r requirements.txt`


# Setting up NVidia

## Install the NVidia Drivers and CUDA Toolkit

By default, Ubuntu uses nouveau drivers, which won't use your NVidia GPU.  To install the NVidia drivers:

```
#Make sure no NVidia or CUDA Drivers are installed
dpkg -l | grep nvidia | awk '{print $2}' | xargs -n1 sudo apt-get purge -y
dpkg -l | grep cuda | awk '{print $2}' | xargs -n1 sudo apt-get purge -y

# Install NVidia Drivers
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get -y update
sudo ubuntu-drivers autoinstall

#NOTE: It is possible to install nvidia-cuda-toolkit here, but I experienced issues with Tensorflow.
#      I found it better to download the files from Nvidia Developer directly and install here.
# NOTE: CUDA 10.2 is available, but Tensorflow 2.1 does not play nice with it.  For now, use CUDA 10.1.

# Install The CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-10-1

# Install cuDNN
# NOTE: This needs to be downloaded - https://developer.nvidia.com/rdp/cudnn-download
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb

# Install tensorrt
# NOTE - This needs to be downloaded - https://developer.nvidia.com/nvidia-tensorrt-6x-download
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda10.1-trt6.0.1.5-ga-20190913/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 libnvinfer-dev=6.0.1-1+cuda10.1 libnvinfer-plugin6=6.0.1-1+cuda10.1
sudo apt-get install -y tensorrt python3-libnvinfer-dev uff-converter-tf

# NOTE: You can delete the .deb files after you are done with them.
```

# Setting up Docker and NVidia Docker

This is the recommended way to run locally.  You can also setup and run through Singularity as described below.

## Install Docker

Docker can be installed in many ways.  On Ubuntu, the easiest is through `sudo snap install docker`.  However, if you plan on using GPUs, it's better to install the apt-get package `docker-ce`.

### Install docker-ce using apt-get

```
# From https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get -y update
sudo apt-get -y install \
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

sudo apt-get -y install docker-ce docker-ce-cli containerd.io
```

### Install Docker using snap

sudo snap install docker

### Add user to the docker group.

Whichever docker installation method you used, you then need to add yourself to the Docker group and logout/login:

```
# From https://docs.docker.com/install/linux/linux-postinstall/
sudo groupadd docker
sudo usermod -aG docker $USER
```

Note that you will need to login/logout or restart for the group addition to take effect.

### Install the NVidia Container Toolkit

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) allows users to build and run GPU accelerated Docker containers.

```
# From https://github.com/NVIDIA/nvidia-docker
export distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### See if it all works

1. Restart the computer.
2. `docker run --gpus all nvidia/cuda nvidia-smi`

You should see all your NVidia GPUs listed.

# Setting up using Singularity

## Install Singularity

There is a [setup script](../setup/singularity.sh) for singularity for Ubuntu 18.04.  Running this script will install:

- [Linux Brew](https://docs.brew.sh/Homebrew-on-Linux)
- [asdf Version Manager](https://asdf-vm.com/)
- [Golang](https://golang.org/)
- [Singularity](https://sylabs.io/docs/)

Hopefully it all works... it was tested on an Ubuntu 18.04 machine, but as there are lots of dependencies your installation may differ.

## Login to Singularity remote (optional)

This section is only needed if you want to build remotely.  You can build locally with either `sudo` or `fakeroot` and not login remotely.

Create an account at [Singularity Cloud](https://cloud.sylabs.io/home) and generate a [token](https://cloud.sylabs.io/auth/tokens).

`singularity remote login` and paste the token you created.


## Build the singularity container

`singularity build --fakeroot image.sif image.def`

## See if it all works

`singularity exec --nv image.sif nvidia-smi`