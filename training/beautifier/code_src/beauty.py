import tensorflow as tf
# import tensorflow_model_optimization as tfmot

def make_discriminator_model(input_shape, path=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation="softmax") )
    if path is not None:
        model.load_weights(path)

    return model

def load_model(cfg):
    # return make_discriminator_model(input_shape=(*cfg["training"]["size"], 3), path=cfg["model"]["beauty"]["path"])
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, alpha=0.35,
                                                                pooling='avg', input_shape=(*cfg["training"]["size"], 3))

    x = base_model.output
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2,  activation="softmax")(x)
    model = tf.keras.Model(base_model.input, outputs)
    # model = tfmot.quantization.keras.quantize_model(model)
    # with tfmot.quantization.keras.quantize_scope():
        # model.load_weights(cfg["model"]["beauty"]["path"])

    model.load_weights(cfg["model"]["beauty"]["path"])
    return model
