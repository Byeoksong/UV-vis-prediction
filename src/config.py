
from dataclasses import dataclass

@dataclass
class Config:
    MAX_LENGTH_SMILES: int = 10  # This will be updated dynamically
    EMBED_DIM: int = 256
    SMILES_ATT_NUM_HEADS: int = 4
    SMILES_ATT_DFF: int = EMBED_DIM * 4
    SMILES_VOCAB_SIZE: int = 50  # This will be updated dynamically
    CHEMICAL_ATT_NUM_HEADS: int = 4
    CHEMICAL_ATT_DFF: int = EMBED_DIM * 4
    CHEMICAL_TRANSFORMER_NUM_LAYERS: int = 3
    PROCESS_ATT_NUM_HEADS: int = 1
    PROCESS_ATT_DFF: int = EMBED_DIM * 4
    PROCESS_TRANSFORMER_NUM_LAYERS: int = 3
    LR: float = 1e-4
    BATCH_SIZE: int = 512
    DATA_PATH: str = "data/"
    SMILES_FILE: str = "input_SMILES.txt"
    ABSORBANCE_FILE: str = "Absorbances.npy"
    MODEL_FILE: str = "UVvis_attention_model.keras"
