from tensorflow.keras import layers


def augment(inputs):
    inputs = layers.RandomBrightness(0.2)(inputs)
    inputs = layers.RandomContrast(0.2)(inputs)
    inputs = layers.RandomTranslation(0.5, 0.5)(inputs)
    inputs = layers.RandomFlip()(inputs)
    inputs = layers.RandomRotation(1)(inputs)

    return inputs
