# pygfunction3 : A prototype implementation of [pygfunction](https://github.com/MassimoCimmino/pygfunction) version 3.0


## What is pygfunction3?

pygfunction3 is an open-source toolbox for the evaluation oof thermal response
factors, or *g*-functions, for fields of geothermal boreholes. *g*-functions
form the basis of many simulation and sizing programs for geothermal heat pump
systems. *g*-Functions are superimposed in time to predict fluid and ground
temperatures in these systems.

This repository is used for the development of version 3 of [pygfunction](https://github.com/MassimoCimmino/pygfunction).
Development is done in a separate repository due to the significant changes in
modelling approaches between the current version of pygfunction and the one used
in this repository. While pygfunction superimposes segment-to-segment finite
line source solutions to obtain segment-by-segment borehole wall temperature
averages from segment-by-segment uniform heat extraction rates, the new
approach in pygfunction superimposes higher-order finite line source solutions
to evaluate borehole wall temperatures at nodes along the boreholes from heat
extraction rates at nodes along the boreholes, assuming polynomial variations
of both quantities along the boreholes.

More details on the method can be found in the journal paper
[(Cimmino, Basquens and Lazzarotto, 2026)](https://www.sciencedirect.com/science/article/pii/S1290072925005071).
The method is also implemented in the open-source Python package [geothermsim](https://github.com/MassimoCimmino/geothermsim).

pygfunction3 and geothermsim have different scopes. pygfunction3 specializes
in the fast evaluation of g-functions.

Example usage can be found in the [notebooks](https://github.com/MassimoCimmino/pygfunction3/notebooks).

## Installation


### Quick start

pygfunction3 uses [uv](https://docs.astral.sh/uv/) to manage dependencies and
the development environment. Install uv by following the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/).

**Developers** - To get the latest version of the code, you can [download the
repository from github](https://github.com/MassimoCimmino/pygfunction3) or clone
the project in a local directory using git:

```
git clone https://github.com/MassimoCimmino/pygfunction3.git
```

Create a virtual environment and install pygfunction3 (in editable mode) along
with its dependencies:
```
uv sync
```

Plotting helpers in `pygfunction3.plot` require matplotlib, which is an optional
extra. Install it with:
```
uv sync --extra plot
```

To install the extra packages needed to run the notebooks (which include
matplotlib), include the `notebooks` dependency group:
```
uv sync --group notebooks
```

Once pygfunction3 is installed, you can verify that it is working properly by
running the [notebooks](notebooks/README.md). Any command can be run inside the
managed environment with `uv run`, e.g. `uv run jupyter lab`.

### Requirements

pygfunction3 was developed using Python 3.14 and is compatible with Python
versions >= 3.12. Dependencies are declared in `pyproject.toml` and pinned in
`uv.lock`. The core runtime dependencies are:
- numpy (>= 2.3.5)
- scipy (>= 1.16.3)
- SecondaryCoolantProps (>= 1.3.0)
- typing_extensions (>= 4.15.0)

The `plot` optional extra provides:
- matplotlib (>= 3.10.6)

The `notebooks` dependency group additionally provides:
- jupyter
- matplotlib (>= 3.10.6)
- pandas (>= 2.3.3)

## License

pygfunction3 is licensed under the terms of the 3-clause BSD-license.
See [pygfunction3 license](LICENSE.md).
