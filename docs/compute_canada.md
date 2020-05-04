# Make a Compute Canada Account

Go to the [Registration Page](https://ccdb.computecanada.ca/account_application) and fill out the form.  You will need the CCI of your supervisor.  Once approved, your account should be provisioned within a day.

# Connecting to Compute Canada

The can check [status](https://status.computecanada.ca/) and view the [list of clusters](https://www.computecanada.ca/research-portal/accessing-resources/available-resources/).

In general, using `graham` or `cedar` is good for GPU jobs.

Connect to the cluster with `ssh <USER>@<CLUSTER>.computecanada.ca`.

# Install Anaconda or Miniconda

`bash <(curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)`

After installation you may need to run `conda init`.  You should see something like the following in the file `~.bashrc`

```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/bgk/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/bgk/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/bgk/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/bgk/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
Then either logout/login or `source ~/.bashrc` to load conda.  You should see `(base)` in your prompt which specifies the Conda environment you are currently using.

Finally, downgrade python to 3.6 for now as it is compatible:

`conda install python=3.6`

# Add your SLURM_ACCOUNT info to your ~./bashrc

`export SLURM_ACCOUNT=def-<SUPERVISOR>`

You can also add envvars for data directories:

```
export DATA_DIR=/scratch/<USER>/datasets/
export OUTPUT_DIR=/scratch/<USER>/results/
export PROJECT_DIR=/project/def-<SUPERVISOR>/<USER>/segmenter/
```

Finally, it's also good to add `module load singularity` so that it's not forgotten.

Then either logout/login or `source ~/.bashrc`.

# Checkout the Git Project on Compute Canada

## Add a Read-Only SSH Key

Github has good [instructions](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account).  You need to:

1. [Generate a new key](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)
2. [Add it to the ssh agent](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent)

You can't install xclip on compute canada servers, so just open the file and copy and paste it.

3. `cat ~/.ssh/id_rsa.pub`  The public key will begin with `ssh-rsa` and end with the e-mail address you used in step 1.
4. Follow the rest of the instructions to [add the key to your account](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)

## Clone the repo

1. `cd /project/def-<SUPERVISOR>/<USER>`
2. `git clone git@github.com:brandongk60/segmenter.git`

## Login to Singularity remote

This section is only needed if you want to build remotely.  You can build locally with either `sudo` or `fakeroot` and not login remotely.

Create an account at [Singularity Cloud](https://cloud.sylabs.io/home) and generate a [token](https://cloud.sylabs.io/auth/tokens).

`singularity remote login` and paste the token you created.

# Build the Singularity image

1. `cd /project/def-<SUPERVISOR>/<USER>/segmenter/`
2. `module load singularity`
3. `singularity build --remote image.sif image.def`.  You will need to create a singularity account and it should guide you through these steps here.

# Upload the dataset

You should have created a [dataset](dataset.md) and processed it.

The dataset should go on the `/scratch` drive.  The scratch drive has faster i/o, but is not persisted.  If the files are stale there for a certain number of weeks, they get deleted.  On your local machine:

1. `cd segmenter/datasets/`
2. `rsync -r --progress --update --delete --exclude '*.png' . <USER>@<CLUSTER>.computecanada.ca:/scratch/<USER>/datasets/`
