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

`conda create -n segmenter python=3.6`

# Install the requirements

`conda activate segmenter && pip install -r requirements.txt`

# View the pydocs

`pydoc -b`
