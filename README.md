## Optical Pooled Screens

This repository contains code and computational tools related to the publication, High-content image-based pooled screens reveal regulators of synaptogenesis.
Raw image data are available on Google Cloud at gs://opspublic-east1/SynaptogenesisOpticalPooledScreen.

For new projects using optical pooled screens, it is highly recommended to use the Github repository accompanying our Nature Protocols paper, Pooled genetic perturbation screens with image-based phenotypes: https://github.com/feldman4/OpticalPooledScreens.

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
