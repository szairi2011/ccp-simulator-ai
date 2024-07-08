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

## Converting and synching .py and .ipynb
Usually it is a good practice to start with an .ipynb notebook to experiment the code by executing one cell at a time with a fine-grained control over different parts of the code.
Once the code is ready, it can be converted to a .py file for a more efficient execution.
To convert the .ipynb to .py file, follow the below steps:

```sh
$ pip install jupytext # Already don for this project
```
To create and keep both files in sync, run below command:

```sh
$ jupytext --set-formats ipynb,py:percent custom_mini_decoder_transformer.py --sync
```
This will create a new .ipynb file from the .py file provided above, i.e. from custom_mini_decoder_transformer.py., and keep them in sync. So, if we modify the newly created .ipynb and save it will update the .py counterpart.

