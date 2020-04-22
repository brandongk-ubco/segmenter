#!/usr/bin/env bash

set -euo pipefail

# FROM: https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps

# Install GO through asdf

# Install homebrew
sudo apt-get -y update
sudo apt-get -y install build-essential curl file git

# Install asdf version manager
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.7.8

# NOTE: You may need to add this to your path as well.
export ASDF_DIR=$(brew --prefix asdf)

# Add asdf go plugin
asdf plugin-add golang https://github.com/kennyp/asdf-golang.git

# Install asdf versions
asdf install

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
wget https://github.com/sylabs/singularity/releases/download/v${SINGULARITY_VERSION}/singularity-${SINGULARITY_VERSION}.tar.gz
tar -xzf singularity-${SINGULARITY_VERSION}.tar.gz

cd singularity
./mconfig
sudo make -C builddir
sudo make -C builddir install
cd -
sudo rm -rf singularity
rm singularity-*.tar.gz
