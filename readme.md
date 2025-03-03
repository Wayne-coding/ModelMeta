# ModelMeta     ISSTA2025

This is the open resposity of the paper "Improving Deep Learning Framework Testing with Model-Level Metamorphic Testing". Here is the structure of the result.



### Description

In this work, we propose ModelMeta, a model-level metamorphic testing method for DL frameworks with four MRs focused on model structure and calculation logic. ModelMeta inserts external structures to generate new models with consistent outputs, increasing interface diversity and detecting bugs without additional MRs. Besides, ModelMeta uses the QR-DQN strategy to guide model generation and then detects bugs from more fine-grained perspectives of training loss, memory usage, and execution time.


If you have any questions, please leave a message here to contact us. 


# Run

## Installation

Ensure you are using Python 3.9 and a Linux-64 platform:

```bash
$ conda create -n ModelMeta python=3.9
$ conda activate ModelMeta
$ pip install -r requirements.txt
```

## Dataset

We provide a few simple datasets`./dataset/`. Due to the large size of other datasets, we will provide a download link upon request. Please contact us to obtain the dataset.

## Usage

### Step 1: Run the master file

```bash
cd ./mindspore_mutation
python main_ms.py
```
```bash
cd ./torch_mutation
python main_torch.py
```
```bash
cd ./onnx_mutation
python main_onnx.py
```
### Step 2: Check Output

Results will be available in the `./mindspore_mutation/results/` ,`./torch_mutation/results/` or `./onnx_mutation/results/` directory.. This folder will contain two files:
- A `.json` file: Contains the log details.
- A `.xlsx` file: Records the results of the process, including coverage, distance, and other relevant metrics.


## Parameter Settings

The parameters for running the mutation tests can be configured in `main_torch.py` , `main_ms.py` or `main_onnx.py`. Below are the adjustable parameters:

- `seed_model`: Name of the model. Options: `resnet`,`UNet`,`vgg16`,`textcnn`.`ssimae`
- `mutation_iterations`: Number of mutation iterations.
- `mutate_times`: Number of epochs for training.
- `batch_size`: Size of the batches for training.

- `mutation_strategy`: Mutation strategy. Options: `'random'`, `'qrdqn'`, `'MCMC'`.



