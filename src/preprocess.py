
import numpy as np
from .data_loader import split_smiles_detailed

def preprocess_data(smiles, weight_smiles, temp_info, config):
    smiles_vocab = set()
    MAX_LENGTH_SMILES = 1
    for i in range(len(smiles)):
        for j in range(len(smiles[i])):
            if MAX_LENGTH_SMILES < len(smiles[i][j]):
                MAX_LENGTH_SMILES = len(smiles[i][j])
            for char in split_smiles_detailed(smiles[i][j]):
                smiles_vocab.add(char)
    smiles_vocab = sorted(list(smiles_vocab))
    smiles_to_index = dict([(char, i + 1) for i, char in enumerate(smiles_vocab)])
    SMILES_VOCAB_SIZE = len(smiles_to_index) + 1

    config.MAX_LENGTH_SMILES = MAX_LENGTH_SMILES
    config.SMILES_VOCAB_SIZE = SMILES_VOCAB_SIZE

    len_max = np.max(list(map(lambda x: len(x), smiles)))
    encoder_array = np.zeros([len(smiles), len_max, config.MAX_LENGTH_SMILES])
    weight_array = np.zeros([len(weight_smiles), len_max])
    temp_info = np.array(temp_info)[:, [0, -1]]

    for i in range(len(smiles)):
        for j in range(len(smiles[i])):
            weight_array[i, j] = weight_smiles[i][j]
            for k, char in enumerate(split_smiles_detailed(smiles[i][j])):
                encoder_array[i, j, k] = smiles_to_index[char]

    return encoder_array, weight_array, temp_info, config
