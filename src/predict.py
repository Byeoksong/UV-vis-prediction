
import tensorflow as tf
from tensorflow import keras
from .model import CustomLearningRateSchedule, RoPE, ReshapeLayer, Expanded_matrix, MaskProcessor

def load_model_for_prediction(model_path):
    return keras.models.load_model(
        model_path,
        custom_objects={
            "CustomLearningRateSchedule": CustomLearningRateSchedule,
            "RoPE": RoPE,
            "ReshapeLayer": ReshapeLayer,
            "Expanded_matrix": Expanded_matrix,
            "MaskProcessor": MaskProcessor,
        }
    )

def predict(model, X_tot, X_weight_tot, X_temp_tot):
    return model.predict([X_tot, X_weight_tot, X_temp_tot])
