## Project creation and venv activation:

The project is created usng the follwing steps:

```sh
## Create the virtual environment
$ python3 -m venv .ccp_simulator_venv

## Activate the virtual environment
$ source .ccp_simulator_venv/Scripts/activate
(.ccp_simulator_venv)

## Deactivate when done with the virtual environment
$ deactivate
```

## Install and setup Jupyter notebook for the current project

To enable Jupyter notebooks for the current project follow below steps:
 Make sure that the virtual environment is activated first

```sh
## Install Jupyter Lab, and extension of Jupyter notebook
$ pip install jupyterlab

## To run in vscode
Open a notebook (i.e. .ipynb file) and select the right Kernel

## To run from a web UI, Start the Jupyter Lab server for the current project
$ jupyter lab
```

## Install packages:

```sh
$ pip install random
```
```sh
$ pip install pandas
```
```sh
$ pip install torch
```
```sh
$ pip install transformers
```




