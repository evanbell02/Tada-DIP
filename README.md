# Tada-DIP: Input-adaptive Deep Image Prior for One-shot 3D Image Reconstruction

This repository contains code to reproduce the main results of the paper "Tada-DIP: Input-adaptive Deep Image Prior for One-shot 3D Image Reconstruction," as presented at the 2025 Asilomar Conf. on Signals, Systems, and Computers.

![Tada-DIP Setup](figs/TadaDIPSetup.png)

### Data and environment set up
The data used for evaluating each of the methods comes from the 2016 AAPM Low Dose CT Grand Challenge Dataset, and is publicly available [here](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715). The conda environment need to run experiments can be created with `conda env create --file environment.yml`, which will create the environment `tada-dip`. The final necessary installation is [LEAP](https://github.com/LLNL/LEAP), which supplies the X-ray projectors.

### Image reconstruction with Tada-DIP
The main results of the paper can reproduced by running `scripts/tada-dip.py`, being sure to set appropriate file paths for the data to be loaded and results to be saved. The three volumes we used for evaluation were `L067`, `L096`, and `L143`. The same script can also be used to run Vanilla DIP experiments with the hyperparameters set as $\alpha=\beta=\gamma=0$.
