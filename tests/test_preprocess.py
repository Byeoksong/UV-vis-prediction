
from src.preprocess import split_smiles_detailed, preprocess_data
from src.config import Config

def test_split_smiles_detailed():
    assert split_smiles_detailed("NaCl") == ["Na", "Cl"]
    assert split_smiles_detailed("CO2") == ["C", "O", "2"]
    assert split_smiles_detailed("H2O") == ["H", "2", "O"]

def test_preprocess_data():
    smiles = [["!", "_C"], ["!", "_N"]]
    weight_smiles = [[1.0, 12.0], [1.0, 14.0]]
    temp_info = [[10, 20], [11, 22]]
    config = Config()

    encoder_array, weight_array, temp_info_processed, updated_config = preprocess_data(smiles, weight_smiles, temp_info, config)

    assert encoder_array.shape == (2, 2, 2)
    assert weight_array.shape == (2, 2)
    assert temp_info_processed.shape == (2, 2)
    assert updated_config.MAX_LENGTH_SMILES == 2
    assert updated_config.SMILES_VOCAB_SIZE > 0
