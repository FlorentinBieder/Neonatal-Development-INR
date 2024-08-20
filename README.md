# Modeling the Neonatal Brain Development Using Implicit Neural Representations
This is the Code Repository for the MICCAI-PRIME 2024 paper
["Modeling the Neonatal Brain Development Using Implicit Neural Representations"](https://arxiv.org/abs/2408.08647).

Check out our [Project Page](https://florentinbieder.github.io/Neonatal-Development-INR/)!


## Installation
We recommend using [miniforge3](https://github.com/conda-forge/miniforge) to 
install the necessary python version in a new mamba environment 
using the `conda_envs.yaml` file, with

    mamba env install -n braindev -f conda_envs.yaml

You can then activate it with

    mamba activate braindev

## Use
You can configure the training or inference using `config.yaml`. The default
values are all set in the `defaults.yaml` file. To run it, use

    python main.py config=config.yaml

You can add additional configurations on the command line to overwrite
the settings from the config files. To run the inference, you need to define a
`start_epoch`, the `log_dir` from the trained model, 
change the `mode` if you want to use a different dataset and 
specify `inference=true`.
The code assumes that you specify the three different `database_*`-csv files for the
e.g. training and inference. In this repo, we provide a sample with three entries.
For inference, we assume that each `subject_id` has scans from two distinct `session_id`s.
We fit the latent vector to one of those scans, and then evaluate the model at the age of the
other scan to compare.

## Acknowledgements & Data
For the implementation of the activations we are grateful that we could rely on the implementation of
[Neural Implicit Segmentation Functions](https://github.com/NILOIDE/Implicit_segmentation/).
Please check out their paper in [MICCAI 2023](https://conferences.miccai.org/2023/papers/466-Paper3205.html)!

The dataset we used was the [dHCP](https://biomedia.github.io/dHCP-release-notes/) dataset (third release).

## Cite
We've published the preprint on arXiv:

```
@article{bieder2024modeling,
  title={Modeling the Neonatal Brain Development Using Implicit Neural Representations},
  author={Bieder, Florentin and Friedrich, Paul and Corbaz, H{\'e}l{\`e}ne and Durrer, Alicia and Wolleb, Julia and Cattin, Philippe C},
  journal={arXiv preprint arXiv:2408.08647},
  year={2024}
}
```
