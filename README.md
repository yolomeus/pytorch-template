# PyTorch Template: PyTorch + Lightning + Hydra

My template for deep learning projects using [PyTorch](https://pytorch.org/),
[PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). It's designed with modularity in mind
by allowing to hot-swap different types of models, datasets, losses and optimization procedures. It also supports
for easy iteration over different hyperparameters while preserving reproducibility.

## Requirements

All requirements are listed in `environment.yml`. I highly recommend using [anaconda](https://www.anaconda.com/) for
installing the dependencies like
so:

```shell
conda env create -f environment.yml -n <my-env-name>
```

Depending on your hardware you might need to change the version of **cudatoolkit** in `environment.yml`. If you don't
want to use an nvidia GPU, you can also just remove this entry.

If you wish to not use [anaconda](https://www.anaconda.com/), you can still check each package's version
in `environment.yml` and install them manually using pip / put them into a `requirements.txt` file.

## Library Responsibilities

<table>
<tr>
<td> 

### PyTorch

Basic building blocks and optimization for deep neural networks:

- [Differentiable math operations](https://pytorch.org/docs/stable/torch.html#math-operations)
- [torch.nn](https://pytorch.org/docs/stable/nn.html) for higher level building blocks
- [Common loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

</td>
<td>
  <img src="res/readme/pytorch.png" width="200" alt="PyTorch" />
</td>
</tr>

<tr>
<td>

### Pytorch Lightning

Wrapper around PyTorch for better code structure / easy use of accelerators:

- [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) for any dataset related
  code
- [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) for
  **training-loop**: encapsulates model, loss and optimizer
- [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) for setting up training (logging,
  callbacks, acceleration)

</td>
<td>
<img src="res/readme/pytorch_lightning.png" width="200" alt="PyTorch Lightning" />
</td>
</tr>

<tr>
<td>

### Hydra

Composable configurations in yaml, also overridable through the command line:

- Any module will be represented as
  [**config / composition of configs**](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/)
- We make hydra responsible for **instantiation**: any component can be swapped by specifying a **config**
- Override parameters or even swap full configs via CLI

</td>
<td>
<img src="res/readme/hydra.jpg" width="200" alt="PyTorch Lightning" />
</td>
</tr>


</table>

## Example: Add a custom model and train it

Let's say you want to introduce a custom PyTorch model to the project that you call the **"Beeg Yoshi MLP"** which
is simply a ridiculously wide MLP.

### 1. Implement your model by subclassing `torch.nn.Module`:

```python
# model/mlp.py
from torch.nn import Module, Linear, Sequential, ReLU


class BeegYoshiMLP(Module):
    """The Beeg Yoshi MLP. It's width will be the input_dim * beeg_factor.
    """

    def __init__(self, input_dim: int, output_dim: int, beeg_factor: int):
        super().__init__()

        h_dim = input_dim * beeg_factor
        self.seq = Sequential(Linear(input_dim, h_dim),
                              ReLU(),
                              Linear(h_dim, output_dim))

    def forward(self, inputs):
        return self.seq(inputs)
```

We've placed this code in `model/mlp.py`. As you will see in the next step, the package structure is mirroring the
config structure. As long as you can reference your class by its **module path** you can also change this structure.
However, I'd encourage you to also mirror package and config structure.

### 2. Add a yaml configuration for your model:

```yaml
# conf/model/beeg_mlp.yaml

# Module path to our class. The _target_ field will be used 
# to reference the class at instantiation and all other entries will be passed to the 
_target_: model.mlp.BeegYoshiMLP

# we will be training on fashion mnist, hence these dimensions
input_dim: 784
output_dim: 10
# Tip: if you want to dynamically change dimensions based on the dataset you could use 
# hydra's interpolation to reference the datamodule config e.g.: 
# input_dim: ${datamodule.input_size}
# check the hydra documentation for details

# the default beeg_factor
beeg_factor: 10
```

Because we've placed our yaml file in `conf/model/` it is now part of the **model** config group. This means we can
select it from the defaults list in our root config file `conf/config.yaml`:

```yaml
# conf/config.yaml
defaults:
  - model: beeg_mlp
  - datamodule: fashion_mnist
  # ...
```

### 3. We're now all set up for training:

```shell
python train.py 
```

Yup that's it. Our config will be automatically assembled and used in `train.py` to instantiate and assemble our
components, then run the training procedure by calling `Trainer.fit()` on our datamodule (atm fashion-mnist).

If you don't want to edit the yaml file you can also override the model and its parameters via CLI:

```shell
python train.py model=beeg_mlp model.beeg_factor=20
```

Hydra also provides a simple interface for gridsearch (there's also auto-ml plugins for hydra, check the docs!):

```shell
python train.py -m model.beeg_factor=1,5,10
```

Hydra will store any logs (including the configuration used) in a dedicated log directory. You can change the directory
pattern at the bottom of `conf/config.yaml`. This is also where e.g. tensorboard logs will end up.

### Summary

This is basically all you need to know to make full use of this template. Each config group and its subgroups are
found in `conf/` and mirrored in the package structure. Just like the model, each part of training can be hot-swapped
as soon as a **config file** and corresponding **python class** exist. This includes: DataModules, callbacks, loggers,
optimizers, metrics and even the general training/optimization procedure (loop).

## Project components

When assembled, the Trainer is composed as follows:

```shell
Trainer
├── cfg.trainer** # trainer args found in config.yaml
├── loop
│   ├── loss
│   ├── optimizer
│   ├── metrics
│   └── model
├── logger
└── callbacks
```

### train.py

This is the main script where training is started. Here, all training components are instantiated and assembled based
on the hydra config.

### Loop (loop.py)

Loop is a `LightningModule` encapsulating **model**, **optimizer**, **loss function** (PyTorch) and **metrics**
(torchmetrics). It is responsible for defining train, validation and test steps (including calling metrics).

### DataModule (datamodule/*)

The [DataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#) stores any code for fetching,
pre-processing and iterating over a dataset by returning [DataLoaders](https://pytorch.org/docs/stable/data.html).

### Logger (logger/loggers.py):

[Logger](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) that will be passed to the
`Trainer`, needs to implement the `Logger` interface.

### Metrics (logger/utils.py)

Convenience class for managing a set of [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/). Used in the
default training loop to update metric state.


