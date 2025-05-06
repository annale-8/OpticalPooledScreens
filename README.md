## Optical Pooled Screens

Analysis resources for the publication *Pooled genetic perturbation screens with image-based phenotypes*. This is an updated and expanded version of the repository for the original 2019 publication [*Optical pooled screens in human cells*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886477/), available [here](https://github.com/feldman4/OpticalPooledScreens_2019).

### Installation (OSX)

Download the repository (e.g., on Github use the green "Clone or download" button, then "Download ZIP").

In Terminal, go to the NatureProtocols project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv
```

If the python3 command isn't available, you might need to specify the full path. E.g., if [Miniconda](https://conda.io/miniconda.html) is installed in the home directory:

```bash
~/miniconda3/bin/python -m venv venv
```

This creates a virtual environment called `venv` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```bash
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

Additionally, if using the CellPose segmentation method, this must be installed in the virtual environment:
```bash
pip install cellpose[gui]
```
