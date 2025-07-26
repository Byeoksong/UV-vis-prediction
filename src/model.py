
import tensorflow as tf
from tensorflow import keras

class CustomLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, initial_lr, peak_lr, tot_steps, n_epoch, **kwargs):
        super(CustomLearningRateSchedule, self).__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.tot_steps = tot_steps
        self.n_epoch = n_epoch

    def __call__(self, step):
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        learning_rate = tf.cond(
            step < self.warmup_steps,
            lambda: tf.cast(warmup_lr, dtype=tf.float32),
            lambda: tf.cast(self.peak_lr, dtype=tf.float32)
        )
        return learning_rate

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
            "peak_lr": self.peak_lr,
            "tot_steps": self.tot_steps,
            "n_epoch": self.n_epoch,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RoPE(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(RoPE, self).__init__(**kwargs)
        self.dim = dim
        self.inv_freq = 1. / (10000 ** (tf.range(0, dim, 2.0) / dim))

    def call(self, inputs_even, inputs_odd, positions=None):
        seq_len = tf.shape(inputs_even)[1]
        if positions is None:
            positions = tf.range(seq_len, dtype=tf.float32)
        freqs = tf.einsum('i,j->ij', positions, self.inv_freq)
        cos = tf.cos(freqs)
        sin = tf.sin(freqs)
        x_rotated_even = inputs_even * cos - inputs_odd * sin
        X_rotated_odd = inputs_even * sin + inputs_odd * cos
        x_rotated = tf.reshape(
            tf.concat([x_rotated_even, X_rotated_odd], axis=-1), (-1, seq_len, self.dim)
        )
        return x_rotated

    def get_config(self):
        return {
            "dim": self.dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MaskProcessor(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskProcessor, self).__init__(**kwargs)

    def call(self, padding_mask):
        padding_mask = tf.cast(padding_mask, tf.float32)
        attention_mask = padding_mask[:, :, tf.newaxis] * padding_mask[:, tf.newaxis, :]
        return attention_mask

def chemical_bert_module(query, key, value, attention_mask, i, config):
    assert config.EMBED_DIM % config.CHEMICAL_ATT_NUM_HEADS == 0
    attention_output, attention_scores = keras.layers.MultiHeadAttention(
        num_heads=config.CHEMICAL_ATT_NUM_HEADS,
        key_dim=config.EMBED_DIM // config.CHEMICAL_ATT_NUM_HEADS,
        name=f"ChemENC_{i}_multiheadattention",
    )(query, value, key, attention_mask=attention_mask, return_attention_scores=True)
    attention_output = keras.layers.Dropout(0.1, name=f"ChemENC_{i}_att_dropout")(attention_output)
    attention_output = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"ChemENC_{i}_att_layernormalization"
    )(value + attention_output)

    ffn = keras.Sequential(
        [
            keras.layers.Dense(config.CHEMICAL_ATT_DFF, activation='silu'),
            keras.layers.Dense(config.EMBED_DIM)
        ],
        name=f"ChemENC_{i}_ffn"
    )
    ffn_output = ffn(attention_output)
    ffn_output = keras.layers.Dropout(0.1, name=f"ChemENC_{i}_ffn_dropout")(ffn_output)
    sequence_output = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"ChemENC_{i}_ffn_layernormalization"
    )(attention_output + ffn_output)
    return sequence_output

def process_bert_module(query, key, value, attention_mask, i, config):
    assert (config.EMBED_DIM) % config.PROCESS_ATT_NUM_HEADS == 0
    attention_output, attention_scores = keras.layers.MultiHeadAttention(
        num_heads=config.PROCESS_ATT_NUM_HEADS,
        key_dim=(config.EMBED_DIM) // config.PROCESS_ATT_NUM_HEADS,
        name=f"ProcENC_{i}_multiheadattention",
    )(query, value, key, attention_mask=attention_mask, return_attention_scores=True)
    attention_output = keras.layers.Dropout(0.1, name=f"ProcENC_{i}_att_dropout")(attention_output)
    attention_output = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"ProcENC_{i}_att_layernormalization"
    )(value + attention_output)
    ffn = keras.Sequential(
        [
            keras.layers.Dense(config.PROCESS_ATT_DFF, activation='silu'),
            keras.layers.Dense(config.EMBED_DIM)
        ],
        name=f"ProcENC_{i}_ffn"
    )
    ffn_output = ffn(attention_output)
    ffn_output = keras.layers.Dropout(0.1, name=f"ProcENC_{i}_ffn_dropout")(ffn_output)
    sequence_output = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"ProcENC_{i}_ffn_layernormalization"
    )(attention_output + ffn_output)
    return sequence_output

class Expanded_matrix(keras.layers.Layer):
    def __init__(self, embedding_dim, *args, **kwargs):
        super(Expanded_matrix, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        expanded_m = tf.expand_dims(inputs, axis=-1)
        outputs = tf.tile(expanded_m, [1, 1, self.embedding_dim])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ReshapeLayer(keras.layers.Layer):
    def __init__(self, shape, *args, **kwargs):
        super(ReshapeLayer, self).__init__(*args, **kwargs)
        self.shape = shape

    def call(self, inputs):
        x_reshaped = tf.reshape(inputs, self.shape)
        return x_reshaped

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shape": self.shape
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_model(config, dim_X, dim_target, X_temp_train_shape):
    input_smiles = keras.layers.Input(shape=dim_X, name='input_smiles')
    input_weight = keras.layers.Input(shape=dim_X[:-1], name='input_weight')
    input_temp = keras.layers.Input(shape=(X_temp_train_shape,), name='input_temp')

    reshape_smiles = ReshapeLayer(shape=(-1, dim_X[1]))
    reshape_process = ReshapeLayer(shape=(-1, dim_X[0], config.EMBED_DIM))
    embedding_layer = keras.layers.Embedding(input_dim=config.SMILES_VOCAB_SIZE, output_dim=config.EMBED_DIM, name='embed_layer')
    embed_weight = Expanded_matrix(config.EMBED_DIM)

    x = input_smiles
    process_padding_mask = keras.layers.Embedding(dim_X[0], config.EMBED_DIM, mask_zero=True, trainable=False)(x[:, :, 0])._keras_mask
    process_attention_mask = MaskProcessor()(process_padding_mask)
    x = reshape_smiles(input_smiles)
    chemical_padding_mask = keras.layers.Embedding(config.SMILES_VOCAB_SIZE, config.EMBED_DIM, mask_zero=True, trainable=False)(x)._keras_mask
    chemical_attention_mask = MaskProcessor()(chemical_padding_mask)

    x = embedding_layer(x)
    x *= tf.math.sqrt(tf.cast(config.EMBED_DIM, tf.float32))
    for i in range(config.CHEMICAL_TRANSFORMER_NUM_LAYERS):
        query = RoPE(config.EMBED_DIM)(x[:, :, 0::2], x[:, :, 1::2])
        key = RoPE(config.EMBED_DIM)(x[:, :, 0::2], x[:, :, 1::2])
        x = chemical_bert_module(query, key, x, chemical_attention_mask, i, config)
    x = x[:, 0, :]

    x = reshape_process(x)
    x_weight = embed_weight(input_weight)
    scaling_weight = keras.layers.Dense(1, activation='softplus', name='WeightScale')(tf.ones((1, 1), dtype=tf.float32))
    x_weight = x_weight * scaling_weight
    x = x * x_weight
    for i in range(config.PROCESS_TRANSFORMER_NUM_LAYERS):
        query = RoPE(config.EMBED_DIM)(x[:, :, 0::2], x[:, :, 1::2])
        key = RoPE(config.EMBED_DIM)(x[:, :, 0::2], x[:, :, 1::2])
        x = process_bert_module(query, key, x, process_attention_mask, i, config)
    x = x[:, 0, :]

    scaling_temp = keras.layers.Dense(X_temp_train_shape, activation='sigmoid', name='TempScale')(tf.ones((1, 1), dtype=tf.float32))
    x_temp = input_temp * scaling_temp
    x = keras.layers.Concatenate(axis=-1)([x, x_temp])
    x = keras.layers.Dense(dim_target[0] * 8, activation='silu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(dim_target[0] * 4, activation='silu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(dim_target[0], activation='softplus')(x)

    model = keras.Model(inputs=[input_smiles, input_weight, input_temp], outputs=[output])
    return model
