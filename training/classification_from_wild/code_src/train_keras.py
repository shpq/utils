from random import random
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
if tf.__version__ == "2.4.0":
    from tensorflow.keras.applications import MobileNetV3Small
    from tensorflow.keras.applications import MobileNetV3Large


quantize_model = tfmot.quantization.keras.quantize_model

physical_devices = tf.config.list_physical_devices('GPU')

for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)

map_name_with_model = {
    "mobilenet_v2": MobileNetV2,
}
if tf.__version__ == "2.4.0":
    map_name_with_model["mobilenet_v3_small"] = MobileNetV3Small
    map_name_with_model["mobilenet_v3_large"] = MobileNetV3Large


def train_keras(cfg, kwargs):
    num_classes = len(np.unique(kwargs["train"].label))
    base_model = map_name_with_model[cfg.model.name](weights='imagenet', include_top=False, alpha=cfg.model.mobilenet_v2.alpha,
                                                     pooling='avg', input_shape=(*cfg.training.img_size, 3))

    filepath = os.path.join(
        cfg.training.checkpoints_path,
        '{epoch:02d}-loss-{val_loss:.2f}.h5'
    )
    os.makedirs(cfg.training.checkpoints_path, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=cfg.training.epoch_stop),
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_acc",
            filepath=filepath,),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=cfg.training.epoch_reduce,
            verbose=1,
        )
    ]
    x = base_model.output
    if float(cfg.model.dropout) > 0:
        x = layers.Dropout(float(cfg.model.dropout))(x)

    outputs = layers.Dense(
        num_classes,  activation="softmax")(x)

    model = keras.Model(base_model.input, outputs)
    if cfg.model.quantize:
        model = quantize_model(model)
    optimizer = keras.optimizers.Adam(learning_rate=cfg.training.lr)
    if cfg.model.pretrained_path:
        if cfg.model.quantize:
            with tfmot.quantization.keras.quantize_scope():
                model = keras.models.load_model(cfg.model.pretrained_path)
        else:
            model = keras.models.load_model(cfg.model.pretrained_path)
        print(cfg.model.pretrained_path + ' loaded')

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    H = model.fit(
        kwargs["dataloaders"]["train"],
        validation_data=kwargs["dataloaders"]["valid"],
        steps_per_epoch=kwargs["train"].shape[0] // cfg.training.batch_size,
        epochs=cfg.training.epochs, callbacks=callbacks,
    )


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, img_dir, csv,  transform=None, batch_size=32,
                 num_classes=2, shuffle=False, size=(224, 224), strong_aug=None):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.csv = csv
        self.cfg = cfg
        if self.cfg.general.custom == "make_feed":
            print("-------------------- FEED --------------------")
            self.bottom = self.csv["bottom"].values
            self.top = self.csv["top"].values

        self.labels = self.csv.label.values
        self.images = self.csv.name.values
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.size = size
        self.transform = transform
        self.strong_aug = strong_aug
        self.on_epoch_end()

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, index):
        X, y = self.__get_data(index)
        return X, y

    def on_epoch_end(self):
        pass

    def __get_data(self, index):
        X = np.empty((self.batch_size, *self.size, 3))
        y = np.empty((self.batch_size))

        for i, id in enumerate(range(index * self.batch_size, (index+1) * self.batch_size)):
            image = np.array(Image.open(os.path.join(
                self.img_dir, self.images[id])))

            if self.cfg.general.custom == "make_feed":
                image = image[int(self.bottom[id]):int(self.top[id]), :, :]

     
            label = int(self.labels[id])
            if self.transform is not None:

                if isinstance(self.transform, dict):

                    if self.strong_aug is not None and label == 1 and random() < self.strong_aug:
                        transform = self.transform[-1]
                        label = 0
                    else:
                        transform = self.transform[label]
                else:
                    transform = self.transform

                image = transform(image=image, size=self.size, mode="keras")

            X[i, ] = image
            y[i] = label

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)
