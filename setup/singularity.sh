#!/usr/bin/env bash

set -euo pipefail

# FROM: https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps

# Install GO through asdf

# Install homebrew
sudo apt-get -y update
sudo apt-get -y install build-essential curl file git
if ! command -v brew; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
fi

brew install gcc
brew install go

# Install singularity dependencies
sudo apt-get -y update
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup

# Install singularity
export SINGULARITY_VERSION=3.5.2 # adjust this as necessary
if [ ! -d singularity ]; then
    if [ ! -f "singularity-${SINGULARITY_VERSION}.tar.gz" ]; then
        wget https://github.com/sylabs/singularity/releases/download/v${SINGULARITY_VERSION}/singularity-${SINGULARITY_VERSION}.tar.gz
    fi
    tar -xzf singularity-${SINGULARITY_VERSION}.tar.gz
fi

cd singularity
./mconfig
make -C builddir
sudo make -C builddir install
cd -
sudo rm -rf singularity
rm singularity-*.tar.gz
