This setup assumes you are running Ubuntu 18.04.  Note that Docker works on Windows, and using WSL you can run Ubuntu commands pretty easily.  However, it's currently not possible to setup Docker to pass through Nvidia drivers, so it's not possible to use GPUs.

# Setting up NVidia

## Install the NVidia Drivers and CUDA Toolkit

By default, Ubuntu uses nouveau drivers, which won't use your NVidia GPU.  To install the NVidia drivers:

```
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get -y update
sudo ubuntu-drivers autoinstall
sudo apt-get -y install nvidia-cuda-toolkit gcc-6
sudo apt-get -y dist-upgrade
```

# Setting up Docker and NVidia Docker

This is the recommended way to run locally.  You can also setup and run through Singularity as described below.

## Install Docker

Docker can be installed in many ways.  On Ubuntu, the easiest is through `sudo snap install docker`.  However, if you plan on using GPUs, it's better to install the apt-get package `docker-ce`.

### Install docker-ce using apt-get

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

sudo apt-get updatesingularity-3.5.2.tar.gz

sudo apt-get install docker-ce docker-ce-cli containerd.io
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

### Install the NVidia Container Toolkit

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) allows users to build and run GPU accelerated Docker containers.

```
# From https://github.com/NVIDIA/nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
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

## Login to Singularity remote

Create an account at [Singularity Cloud](https://cloud.sylabs.io/home) and generate a [token](https://cloud.sylabs.io/auth/tokens).

`singularity remote login` and paste the token you created.

`singularity build --remote image.sif image.def`

## Build the singularity container


## See if it all works