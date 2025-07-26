# Transformer-based UV-vis Spectra Prediction Model

This repository contains the code for a transformer-based model that predicts UV-vis spectra from the synthesis conditions of nanocrystals. This work is part of a study currently under preparation.

## Project Structure

```
.
├── data/                  # Data files
│   ├── Absorbances.npy
│   ├── input_SMILES.txt

├── notebooks/             # Jupyter notebook (for reference)
│   └── UV-vis_prediction.ipynb
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading utilities
│   ├── model.py           # Model definition
│   ├── predict.py         # Prediction script
│   └── train.py           # Training script
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_preprocess.py
├── .devcontainer/         # Development Container configuration
│   ├── devcontainer.json
│   └── Dockerfile
├── .gitignore
├── main.py                # Main script to run the pipeline
├── README.md
└── requirements.txt       # Python dependencies
```

## Requirements

* Python 3.11+
* TensorFlow
* Keras
* NumPy
* Pymatgen
* RDKit
* pytest (for testing)

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Development Environment (Dev Container)

This project includes a Dev Container configuration for a consistent and isolated development environment.

### Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop)
*   [Visual Studio Code](https://code.visualstudio.com/)
*   [Remote - Containers extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started with Dev Container

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Byeoksong/UV-vis_prediction.git
    cd UV-vis_prediction
    ```
2.  **Open in VS Code:**
    Open the project folder in VS Code. You will be prompted to "Reopen in Container". Click this button.
    If you are not prompted, open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P) and select "Remote-Containers: Reopen in Container".
3.  **Wait for setup:**
    VS Code will build the Docker image (if not already built) and set up the development container. This may take a few minutes on the first run.
4.  **Start developing:**
    Once the container is ready, you will have a fully configured environment with all dependencies installed. You can then run `python main.py` or `pytest` directly in the integrated terminal.

## Usage

To run the full pipeline (data preprocessing, model training, prediction, and performance evaluation), execute the `main.py` script:

```bash
python main.py
```

### Configuration

All configurable parameters, such as file paths, model hyperparameters, and training settings, are located in `src/config.py`.

### Running Tests

To run the unit tests, use `pytest`:

```bash
pytest
```

## Download Model Parameters

You can download the pre-trained model parameters from the following link (~218 MB):
[Google Drive link](https://drive.google.com/file/d/1NSEOnLpVyAAKFDTYcMOjRVut4TOdGHO2/view?usp=sharing)

## License

This project is licensed under the MIT License.

## Citation

This work is currently under preparation. Citation information will be updated upon publication.