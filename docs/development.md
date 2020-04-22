# Install Git lfs

`curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
`sudo apt-get install git-lfs`
`git lfs install`

# Install VSCode

`sudo snap install --classic code`

# Install NVidia Dependencies

Do we need these?
- TensorRT
- cuDNN
- PyCUDA

`https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html`
`pip install 'pycuda>=2019.1.1'`

# View the pydocs

`pydoc -b`

# Optional Installs

ZSH - ` sudo apt-get install -y zsh && sudo usermod -s /usr/bin/zsh $(whoami)`
Oh My ZSH - `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"`
