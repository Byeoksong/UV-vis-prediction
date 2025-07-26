
import numpy as np
from pymatgen.core.periodic_table import Element

def split_smiles_detailed(smiles):
    result = []
    len_smiles = len(smiles)
    n = 0
    while n < len_smiles:
        if smiles[n].isalpha():
            if n+1 < len_smiles:
                if smiles[n].isupper() and smiles[n+1].islower():
                    try:
                        atom = Element(smiles[n:n+2])
                        result.append(smiles[n:n+2])
                        n += 2
                    except:
                        result.append(smiles[n])
                        result.append(smiles[n+1])
                        n += 2
                else:
                    result.append(smiles[n])
                    n += 1
            else:
                result.append(smiles[n])
                n += 1
        else:
            result.append(smiles[n])
            n += 1
    return result

def load_data(config):
    smiles = []
    weight_smiles = []
    temp_info = []
    idx_train = []
    idx_test = []
    idx_tot = []
    with open(config.DATA_PATH + config.SMILES_FILE, 'r') as f:
        i = 0
        while True:
            tmp = f.readline()
            if len(tmp) == 0:
                break
            tmp = tmp.split()
            temp_info.append([float(x) for x in tmp[-5:-1]])
            smiles.append(['!'] + list(map(lambda x: '_' + x, tmp[:-5][::2])))
            weight_smiles.append([1.0] + [float(x) for x in tmp[:-5][1::2]])
            if tmp[-1] == 'T':
                idx_train.append(i)
                idx_tot.append(i)
            elif tmp[-1] == 'F':
                idx_test.append(i)
                idx_tot.append(i)
            i += 1

    target = np.load(config.DATA_PATH + config.ABSORBANCE_FILE)[:, :, 1]

    return smiles, weight_smiles, temp_info, target, idx_train, idx_test, idx_tot
