import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
import os


def _upsample(filters, size, apply_dropout=False):
    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.random_normal_initializer(0., 1.)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def build():

    #   ENCODER

    backbone = MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    layer_names = [
        'activation',    # 112
        're_lu_2',         # 56
        'activation_1',    # 28
        'activation_11',    # 14
        'activation_17',    # 7
    ]
    backbone_outputs = [backbone.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=backbone.input, outputs=backbone_outputs)

    down_stack.trainable = False

    #   DECODER

    up_stack = [
        _upsample(128, 3),
        _upsample(64, 3),
        _upsample(32, 3),
        _upsample(16, 3),
    ]

    # MODEL

    inputs = tf.keras.layers.Input(shape=[224, 224, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding='same')

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()
    model_label = os.path.basename(__file__)
    return model, model_label
