import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from src.config import Config
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.train import train_model
from src.predict import load_model_for_prediction, predict

def main():
    config = Config()

    # Load data
    smiles, weight_smiles, temp_info, target, idx_train, idx_test, idx_tot = load_data(config)

    # Preprocess data
    encoder_array, weight_array, temp_info, config = preprocess_data(smiles, weight_smiles, temp_info, config)

    # Split data
    X_train = tf.cast(encoder_array[idx_train], tf.float32)
    X_weight_train = tf.cast(weight_array[idx_train], tf.float32)
    X_temp_train = tf.cast(temp_info[idx_train], tf.float32)
    y_train = target[idx_train]

    X_test = tf.cast(encoder_array[idx_test], tf.float32)
    X_weight_test = tf.cast(weight_array[idx_test], tf.float32)
    X_temp_test = tf.cast(temp_info[idx_test], tf.float32)
    y_test = target[idx_test]

    X_tot = tf.cast(encoder_array[idx_tot], tf.float32)
    X_weight_tot = tf.cast(weight_array[idx_tot], tf.float32)
    X_temp_tot = tf.cast(temp_info[idx_tot], tf.float32)
    y_tot = target[idx_tot]

    dim_X = encoder_array.shape[1:]
    dim_target = target.shape[1:]

    # Build model
    model = build_model(config, dim_X, dim_target, X_temp_train.shape[-1])
    model.summary()

    # Train model
    train_model(model, X_train, X_weight_train, X_temp_train, y_train, X_test, X_weight_test, X_temp_test, y_test, config)

    # Load trained model and predict
    trained_model = load_model_for_prediction(config.MODEL_FILE)
    predictions = predict(trained_model, X_test, X_weight_test, X_temp_test)
    
    # Calculate and print MSE
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error on the total dataset: {mse}")

if __name__ == "__main__":
    main()