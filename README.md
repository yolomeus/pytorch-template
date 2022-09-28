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