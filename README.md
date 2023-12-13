# Investigating the intrinsic top-down dynamics of deep generative models

### User guide
The code reports the main experiments of Tausani et al., 2023. The code is thought to be used with Google Colab (check [here](DBN_generation.ipynb) for example experiments) using Google Drive (you can find an example [here](https://drive.google.com/drive/folders/1_6f8UFZx2LGFVHqW3wLaBREnlGk069Kc?usp=drive_link)) for storage.

The code needs three `json` configuration files. `cparams.json`,`lparams-DatasetName.json` (e.g. `lparams-mnist.json`)
* `cparams.json` contains configuration parameters: the dataset, the number of layers and others (fixed) parameters
* `lparams-mnist.json` contains the hyper-parameters used to fit the DBN for the MNIST dataset;
* `lparams-celeba_bw.json` contains hyper-parameters to fit the DBN for the CelebA dataset

Examples of the configuration files used in the simulations are listed below.

##### `cparams.json`
````json
{
	"DATASET_ID"       : "MNIST",
	"READOUT"          : false,
	"RUNS"             : 1,
	"LAYERS"           : 3,
	"ALG_NAME"         : "i",
	"NUM_DISCR"        : false,
	"INIT_SCHEME"      : "normal"
}
````

##### `lparams-mnist.json`
```json
{
	"BATCH_SIZE"       : 128,
	"EPOCHS"           : 100,
	"INIT_MOMENTUM"    : 0.5,
	"FINAL_MOMENTUM"   : 0.9,
	"LEARNING_RATE"    : 0.01,
	"WEIGHT_PENALTY"   : 1e-4
}
````

##### `lparams-celeba_bw.json`
```json
{
	"BATCH_SIZE"       : 128,
	"EPOCHS"           : 100,
	"INIT_MOMENTUM"    : 0.5,
	"FINAL_MOMENTUM"   : 0.9,
	"LEARNING_RATE"    : 0.01,
	"WEIGHT_PENALTY"   : 1e-4
}
````


