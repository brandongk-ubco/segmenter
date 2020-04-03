# Make a Compute Canada Account

Go to the [Registration Page](https://ccdb.computecanada.ca/account_application) and fill out the form.  You will need the CCI of your supervisor.  Once approved, your account should be provisioned within a day.

# Connecting to Compute Canada

The can check [status](https://status.computecanada.ca/) and view the [list of clusters](https://www.computecanada.ca/research-portal/accessing-resources/available-resources/).

In general, using `Graham` or `Cedar` is good for GPU jobs.

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

# Add your SLURM_ACCOUNT info to your ~./bashrc

`export SLURM_ACCOUNT=def-<SUPERVISOR>`

Then either logout/login or `source ~/.bashrc`.

# Checkout the Git Project on Compute Canada

## Add a Read-Only SSH Key

Github has good (instructions)[https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account].  You need to:

1. (Generate a new key)[https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key]
2. (Add it to the ssh agent)[https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent]

You can't install xclip on compute canada servers, so just open the file and copy and paste it.

3. `cat ~/.ssh/id_rsa.pub`  The public key will begin with `ssh-rsa` and end with the e-mail address you used in step 1.
4. Follow the rest of the instructions to (add the key to your account)[https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account]

## Clone the repo

1. `cd /project/def-<SUPERVISOR>/<USER>`
2. `git clone git@github.com:brandongk60/segmenter.git`

# Setup the NVIDIA Drivers on the cluster

## Discover the NVIDIA Driver version of the cluster

In the cloned repo, there is a file `nvidia.sh`.  We need to schedule that file as a job to view the current nvidia driver version of the cluster.

1. `cd /project/def-<SUPERVISOR>/<USER>/segmenter`
2. `sbatch nvidia.sh`  This should schedule a job; you should see output like `Submitted batch job 29940889`
3. Wait for the job to complete.  You can see the currently scheduled jobs by `squeue -u $USER`.  If the job list is empty, the job has already completed.
4. `cat ~/nvidia-smi.log`.  You should see output like (where the driver version in this example is `418.87.00`):

```
Sat Apr  4 13:14:20 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
| N/A   34C    P0    25W / 250W |      0MiB / 12198MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Extract the nvidia driver version

Hopefully the .run file for NVidia Driver version exists in this repo already.  If it does not, you will need to find it and download it.  Once it's in the repo:

1. `cd /project/def-<SUPERVISOR>/<USER>/segmenter/nvidia-drivers`
2. `./extract_nvdriver.sh <VERSION> ~/nvidiadriver`
