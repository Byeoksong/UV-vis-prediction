
import tensorflow as tf
from tensorflow import keras
from .model import CustomLearningRateSchedule

def train_model(model, X_train, X_weight_train, X_temp_train, y_train, X_test, X_weight_test, X_temp_test, y_test, config):
    lr_schedule = CustomLearningRateSchedule(
        warmup_steps=5000,
        initial_lr=0.0,
        peak_lr=config.LR,
        tot_steps=tf.math.ceil(len(X_train) / config.BATCH_SIZE),
        n_epoch=100000
    )

    opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, weight_decay=0.01)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size', name='mean_squared_error'),
        metrics=['accuracy'],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config.MODEL_FILE, monitor='val_loss', verbose=2, save_best_only=True, mode='min', initial_value_threshold=0.0003)

    history = model.fit(x=[X_train, X_weight_train, X_temp_train], y=y_train, epochs=30000, batch_size=config.BATCH_SIZE, verbose=0, validation_data=([X_test, X_weight_test, X_temp_test], y_test), callbacks=[checkpoint])

    return history
