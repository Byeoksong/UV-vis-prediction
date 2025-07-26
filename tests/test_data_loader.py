

import os
import numpy as np
from src.data_loader import load_data
from src.config import Config

def test_load_data():
    # Create dummy data files for testing
    os.makedirs("temp_data", exist_ok=True)
    with open("temp_data/input_SMILES.txt", "w") as f:
        f.write("C 1.0 O 2.0 10 20 30 40 T\n")
        f.write("N 3.0 H 4.0 11 22 33 44 F\n")

    np.save("temp_data/Absorbances.npy", np.random.rand(2, 10, 2))

    config = Config()
    config.DATA_PATH = "temp_data/"

    smiles, weight_smiles, temp_info, target, idx_train, idx_test, idx_tot = load_data(config)

    assert len(smiles) == 2
    assert len(weight_smiles) == 2
    assert len(temp_info) == 2
    assert len(idx_train) == 1
    assert len(idx_test) == 1
    assert len(idx_tot) == 2
    assert target.shape == (2, 10)

    # Clean up dummy files
    os.remove("temp_data/input_SMILES.txt")
    os.remove("temp_data/Absorbances.npy")
    os.rmdir("temp_data")

