# PyTorch Template: PyTorch + Lightning + Hydra

My template for deep learning projects using [PyTorch](https://pytorch.org/),
[PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). It's designed to be as modular as
by allowing to hot-swap different types of models, datasets, losses and optimization procedures. It also allows
for easy iteration over different hyperparameters while keeping reproducibility in mind.

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

- Differentiable math operations
- `torch.nn` for higher level building blocks
- common losses

</td>
<td>
  <img src="res/readme/pytorch.png" width="200" alt="PyTorch" />
</td>
</tr>

<tr>
<td>

### Pytorch Lightning

Wrapper around PyTorch for better code structure / easy use of accelerators:

- `DataModule` for any dataset related code
- `LightningModule` for **training-loop**: encapsulates model, loss and optimizer
- `Trainer` for setting up training (logging, callbacks, acceleration)

</td>
<td>
<img src="res/readme/pytorch_lightning.png" width="200" alt="PyTorch Lightning" />
</td>
</tr>

<tr>
<td>

### Hydra

Composable configurations in yaml, also overridable through the command line:

- Any module will be represented as **config / composition of configs**
- We make hydra responsible for **instantiation**: any component can be swapped by specifying a **config**
- Override parameters or even swap full configs via CLI

</td>
<td>
<img src="res/readme/hydra.jpg" width="200" alt="PyTorch Lightning" />
</td>
</tr>


</table>

## Example: How to add a module

Let's say we want to introduce a custom PyTorch model to our project.

## Overall project structure