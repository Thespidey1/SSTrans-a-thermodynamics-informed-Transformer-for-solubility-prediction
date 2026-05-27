# SSTrans-a-thermodynamics-informed-Transformer-for-solubility-prediction
Three-pronged evaluation framework for generalizability tests

This repository contains the code of SSTrans and those used for model development.
The required data files for running the scripts are available on Zenodo:

[https://doi.org/10.5281/zenodo.20392964](https://doi.org/10.5281/zenodo.20392964)
## Requirements for Environment
To run the scripts provided in this repository, you'll need the following Python libraries:

* absl-py==2.3.1
* aimsim_core==2.2.3
* aiohappyeyeballs==2.6.1
* aiohttp==3.13.2
* aiosignal==1.4.0
* alabaster==0.7.16
* alembic==1.16.5
* anyio==4.12.1
* argon2-cffi==25.1.0
* argon2-cffi-bindings==25.1.0
* arrow==1.4.0
* astartes==1.3.3
* asttokens==3.0.1
* astunparse==1.6.3
* async-lru==2.0.5
* async-timeout==5.0.1
* atomInSmiles==1.0.2
* attrs==25.4.0
* babel==2.18.0
* beautifulsoup4==4.14.3
* bleach==6.2.0
* blinker==1.9.0
* cairocffi==1.7.1
* CairoSVG==2.8.2
* certifi==2025.8.3
* cffi==2.0.0
* charset-normalizer==3.4.3
* choreographer==1.2.0
* click==8.1.8
* cloudpickle==3.1.2
* colorama==0.4.6
* colorlog==6.10.1
* comm==0.2.3
* contourpy==1.3.0
* cssselect2==0.8.0
* cycler==0.12.1
* debugpy==1.8.20
* decorator==5.2.1
* defusedxml==0.7.1
* dgl==2.2.1
* dill==0.4.0
* docstring_parser==0.18.0
* docutils==0.21.2
* et_xmlfile==2.0.0
* exceptiongroup==1.3.0
* executing==2.2.1
* fastjsonschema==2.21.2
* fastprop==1.2.1
* filelock==3.18.0
* Flask==3.1.3
* flatbuffers==25.9.23
* fonttools==4.59.2
* fqdn==1.5.1
* freetype-py==2.5.1
* frozenlist==1.8.0
* fsspec==2025.7.0
* future==1.0.0
* gast==0.6.0
* google-pasta==0.2.0
* greenlet==3.2.4
* grpcio==1.76.0
* h11==0.16.0
* h5py==3.14.0
* httpcore==1.0.9
* httpx==0.28.1
* huggingface-hub==0.34.4
* hyperopt==0.2.7
* idna==3.10
* imagesize==1.5.0
* importlib_metadata==8.7.0
* importlib_resources==6.5.2
* iniconfig==2.1.0
* ipykernel==6.31.0
* ipython==8.18.1
* ipywidgets==8.1.8
* isoduration==20.11.0
* itsdangerous==2.2.0
* jedi==0.19.2
* Jinja2==3.1.4
* joblib==1.5.1
* json5==0.14.0
* jsonpointer==3.0.0
* jsonschema==4.25.1
* jsonschema-specifications==2025.9.1
* jupyter==1.1.1
* jupyter-console==6.6.3
* jupyter-events==0.12.1
* jupyter-lsp==2.3.1
* jupyter_client==8.6.3
* jupyter_core==5.8.1
* jupyter_server==2.18.2
* jupyter_server_terminals==0.5.4
* jupyterlab==4.5.7
* jupyterlab_pygments==0.3.0
* jupyterlab_server==2.28.0
* jupyterlab_widgets==3.0.16
* kaleido==1.1.0
* keras==3.10.0
* kiwisolver==1.4.7
* lark==1.3.1
* libclang==18.1.1
* lightgbm==4.6.0
* lightning==2.5.6
* lightning-utilities==0.15.2
* llvmlite==0.43.0
* logistro==2.0.0
* lxml==6.0.2
* lz4==4.4.4
* Mako==1.3.10
* Markdown==3.9
* markdown-it-py==3.0.0
* MarkupSafe==3.0.3
* matplotlib==3.9.4
* matplotlib-inline==0.2.1
* mdurl==0.1.2
* mhfp==1.9.6
* mistune==3.2.1
* ml_dtypes==0.5.3
* mordredcommunity==2.0.6
* mpmath==1.3.0
* multidict==6.7.0
* multiprocess==0.70.18
* mypy_extensions==1.1.0
* namex==0.1.0
* narwhals==2.9.0
* nbclient==0.10.2
* nbconvert==7.17.1
* nbformat==5.10.4
* nest-asyncio==1.6.0
* networkx==3.2.1
* nltk==3.9.2
* notebook==7.5.6
* notebook_shim==0.2.4
* numba==0.60.0
* numpy==1.23.5
* openpyxl==3.1.5
* opt_einsum==3.4.0
* optree==0.17.0
* optuna==4.5.0
* optuna-integration==4.6.0
* orjson==3.11.4
* overrides==7.7.0
* packaging==25.0
* padelpy==0.1.14
* pandas==2.3.1
* pandas_flavor==0.7.0
* pandocfilters==1.5.1
* parso==0.8.5
* pillow==11.3.0
* platformdirs==4.4.0
* plotly==6.3.1
* pluggy==1.6.0
* polars==1.35.1
* polars-runtime-32==1.35.1
* prometheus_client==0.25.0
* prompt_toolkit==3.0.52
* propcache==0.4.1
* protobuf==6.33.0
* psutil==7.1.0
* pure_eval==0.2.3
* py4j==0.10.9.9
* pyarrow==21.0.0
* pycairo==1.28.0
* pycparser==2.23
* Pygments==2.19.2
* pyparsing==3.2.3
* pytest==8.4.2
* pytest-timeout==2.4.0
* python-dateutil==2.9.0.post0
* python-json-logger==4.0.0
* pytorch-lightning==2.5.6
* pytz==2025.2
* pywin32==311
* pywinpty==3.0.3
* PyYAML==6.0.2
* pyzmq==27.1.0
* rdkit==2023.3.3
* referencing==0.36.2
* regex==2025.7.34
* reportlab==4.4.10
* requests==2.32.4
* rfc3339-validator==0.1.4
* rfc3986-validator==0.1.1
* rfc3987-syntax==1.1.0
* rich==14.2.0
* rlPyCairo==0.4.0
* rpds-py==0.27.1
* safetensors==0.6.2
* scikit-learn==1.4.0
* scikit-plot==0.3.7
* scipy==1.13.1
* Send2Trash==2.1.0
* shap==0.49.1
* simplejson==3.20.2
* six==1.17.0
* slicer==0.0.8
* snowballstemmer==3.0.1
* soupsieve==2.8.3
* Sphinx==7.4.7
* sphinxcontrib-applehelp==2.0.0
* sphinxcontrib-devhelp==2.0.0
* sphinxcontrib-htmlhelp==2.1.0
* sphinxcontrib-jsmath==1.0.1
* sphinxcontrib-qthelp==2.0.0
* sphinxcontrib-serializinghtml==2.0.0
* SQLAlchemy==2.0.44
* stack-data==0.6.3
* svglib==1.6.0
* sympy==1.13.3
* tabulate==0.9.0
* tensorboard==2.20.0
* tensorboard-data-server==0.7.2
* tensorboardX==2.6.5
* termcolor==3.1.0
* terminado==0.18.1
* threadpoolctl==3.6.0
* tinycss2==1.4.0
* tokenizers==0.21.4
* tomli==2.3.0
* torch==2.8.0+cu129
* torch-geometric==2.6.1
* torchdata==0.11.0
* torchmetrics==1.8.2
* torchvision==0.23.0+cu129
* tornado==6.5.5
* tqdm==4.67.1
* traitlets==5.14.3
* transformers==4.55.2
* typed-argument-parser==1.11.0
* typing-inspect==0.9.0
* typing_extensions==4.15.0
* tzdata==2025.2
* uri-template==1.3.0
* urllib3==2.5.0
* wcwidth==0.5.3
* webcolors==24.11.1
* webencodings==0.5.1
* websocket-client==1.9.0
* Werkzeug==3.1.3
* widgetsnbextension==4.0.15
* wrapt==2.0.0
* xarray==2024.7.0
* yarl==1.22.0
* zipp==3.23.0

To install required packages, use the command: `pip install <package_name>==<version_number>`. For example, `pip install rdkit==2023.3.3`. The entire requirements can be installled directly through 'pip install -r requirements.txt'.

## Models
The full SSTrans architecture is implemented as the main model class in `models.py`.  
Users can customize the model within the PyTorch framework for training, testing, and prediction tasks.  
Alternatively, you may also follow the detailed usage instructions provided below.

## Predict
Before prediction, please prepare the input file `Smiles_for_pre.xlsx`, which should contain the solute SMILES, solvent SMILES, and temperature information. An example input file is provided in this repository.

The code for loading the trained models and performing prediction is provided in `Predict.py`.

A more detailed step-by-step instruction is available in `demo.ipynb`, which can be run directly.

The trained model parameter files:

- `best_model1.pth`
- `best_model2.pth`
- `best_model3.pth`
- `best_model4.pth`
- `best_model5.pth`

are available on Zenodo. Please download these files and place them in the `Predict` directory before running the prediction code.

## Train
The data used for model construction and pretraining have been pre-tokenized and saved as `.pt` files to accelerate training.

The scripts for training, validation, testing, and pretraining are provided in `Training_validation_test.py`.

You can modify `config.yaml` to adjust:

- Model configurations
- Loss functions
- Hyperparameters
- Different stages of the model development workflow

Please note that leaving the pretraining path empty means that no pretrained model will be used. Moreover, please change the normalizer to 'valid stage' when making predictions on validation sets.

Before training, download the required `.pt` data files from Zenodo and place them in the `Train` directory.

You can directly run `demo.ipynb` to start training and check whether all required files are available.
