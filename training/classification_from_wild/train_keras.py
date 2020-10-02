import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import TrainConfig, create_folder
import tensorflow_model_optimization as tfmot
import itertools
from random import random


quantize_model = tfmot.quantization.keras.quantize_model

physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)
map_name_with_model = {
    "mobilenet_v2": MobileNetV2,
}


def train_keras(FLAGS, kwargs):
    num_classes = len(np.unique(kwargs["train"].label))
    base_model = map_name_with_model[FLAGS.pretrained](weights='imagenet', include_top=False,
                                                       pooling='avg', input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    filepath = os.path.join(
        TrainConfig.checkpoints_folder,
        FLAGS.csv,
        f"{FLAGS.csv}-{FLAGS.pretrained}" + '-{epoch:02d}-loss-{val_loss:.2f}.h5'
    )
    create_folder(filepath)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6),
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_acc",
            filepath=filepath,),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=4,
            verbose=1,
        )
    ]
    x = base_model.output
    if float(FLAGS.dropout) > 0:
        x = layers.Dropout(float(FLAGS.dropout))(x)

    outputs = layers.Dense(
        num_classes,  activation="softmax")(x)

    model = keras.Model(base_model.input, outputs)
    if FLAGS.quantize:
        model = quantize_model(model)
    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr)
    if FLAGS.saved != '-':
        model = keras.models.load(FLAGS.saved)
        print(FLAGS.saved + ' loaded')

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    H = model.fit(
        kwargs["dataloaders"]["train"],
        validation_data=kwargs["dataloaders"]["valid"],
        steps_per_epoch=kwargs["train"].shape[0] // FLAGS.batch_size,
        epochs=FLAGS.epoch, callbacks=callbacks,
    )

    name = os.path.join(
        TrainConfig.checkpoints_folder,
        FLAGS.csv,
        "{}.h5".format(
            FLAGS.csv,

        ))

    model.save(name)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, csv,  transform=None, batch_size=32,
                 num_classes=2, shuffle=False, x=600, y=600, strong_aug=None):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.csv = csv
        self.labels = self.csv.label.values
        self.images = self.csv.name.values
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x = x
        self.transform = transform
        self.strong_aug = strong_aug
        self.y = y
        self.on_epoch_end()

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, index):
        X, y = self.__get_data(index)
        return X, y

    def on_epoch_end(self):
        pass

    def __get_data(self, index):
        X = np.empty((self.batch_size, self.x, self.y, 3))
        y = np.empty((self.batch_size))

        for i, id in enumerate(range(index * self.batch_size, (index+1) * self.batch_size)):
            image = np.array(Image.open(os.path.join(
                self.img_dir, self.images[id])))
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

                image = transform(image=image, x=self.x,
                                  y=self.y, mode="keras")

            X[i, ] = image
            y[i] = label

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)
