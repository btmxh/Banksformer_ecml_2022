import tensorflow as tf
import time
import keras
from keras import layers

from .transformer_core import *


def decoder_layer(
    d_model: int, num_heads: int, dff: int, rate=0.1, epsilon=1e-6, suffix=""
) -> keras.Model:
    x = layers.Input(shape=(None, d_model))
    mask = layers.Input(shape=(None, None, None))

    y = layers.MultiHeadAttention(num_heads, d_model)(
        x, x, x, attention_mask=mask, use_causal_mask=True
    )
    y = layers.Dropout(rate)(y)
    y = layers.LayerNormalization(epsilon=epsilon)(y + x)

    z = point_wise_feed_forward_network(d_model, dff)(y)
    z = layers.Dropout(rate)(y)
    z = layers.LayerNormalization(epsilon=epsilon)(z + y)

    return keras.Model([x, mask], z, name="DecLayer" + suffix)


def decoder(
    num_layers: int,
    d_model: int,
    num_heads: int,
    dff: int,
    max_pe: int,
    inp_dim: int,
    rate=0.1,
    epsilon=1e-6,
) -> keras.Model:
    x = layers.Input(shape=(None, inp_dim))
    mask = layers.Input(shape=(None, None, None))

    y = layers.Dense(dff, activation="relu")(x)
    y = layers.Dense(d_model, activation="relu")(y)

    pos_enc = positional_encoding(max_pe, d_model)
    y = layers.Lambda(lambda y: y + pos_enc[:, : y.shape[1], :])(y)
    y = layers.Dropout(rate)(y)
    for i in range(num_layers):
        y = decoder_layer(
            d_model, num_heads, dff, rate=rate, epsilon=epsilon, suffix=str(i)
        )([y, mask])

    return keras.Model([x, mask], y, name="Decoder")


def Transformer(
    num_layers_enc: int,
    num_layers_dec: int,
    d_model: int,
    num_heads: int,
    dff: int,
    maximum_position_encoding: int,
    inp_dim: int,
    config,
    rate=0.1,
    epsilon=1e-6,
) -> keras.Model:
    tar = layers.Input(shape=(None, inp_dim), name="x")
    mask = layers.Input(shape=(None, None, None), name="mask")
    tar_inp = tar[:, :-1]
    tar_out = tar[:, 1:]
    y = decoder(
        num_layers_dec,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
        inp_dim,
        rate=rate,
        epsilon=epsilon,
    )([tar_inp, mask])
    y = layers.Dense(d_model, name="FinalLayer")(y)

    outputs = []

    for name in config["ORDER"]:
        dim = config["FIELD_DIMS_NET"][name]
        acti = config["ACTIVATIONS"].get(name, None)
        out = layers.Dense(dim, activation=acti, name=name)(y)
        outputs.append(out)

        st = config["FIELD_STARTS_IN"][name]
        end = st + config["FIELD_DIMS_IN"][name]
        to_add = tar_out[:, :, st:end]
        y = layers.Concatenate()([y, to_add])

    return keras.Model([tar, mask], outputs, name="Transformer")


def transformer_feedforward(transformer: keras.Model, x, mask, order):
    dec_output = transformer.get_layer("Decoder")([x, mask])
    final_output = transformer.get_layer("FinalOutput")(dec_output)
    pred_values = {}
    for key in order:
        pred_values[key] = transformer.get_layer(key)(final_output)
    pass


def masked_loss(loss_fn):
    def masked(y_true, y_pred):
        mask = tf.not_equal(tf.reduce_sum(y_true, axis=2), 0)
        mask = tf.cast(mask, dtype=y_true.dtype)
        raw_loss = loss_fn(y_true, y_pred)
        raw_loss *= mask
        return tf.reduce_sum(raw_loss) / tf.reduce_sum(mask)

    return masked
