import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, AveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from code_src.FRNTLU import *


class ConvLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride, activation=None, pad="same", transpose=False):
        super().__init__()

        if transpose:
            conv = tf.keras.layers.Conv2DTranspose
        else:
            conv = Conv2D

        if pad == "refl":
            layers = [
                ReflectionPad2d(kernel_size // 2),
                conv(filters=out_channels, kernel_size=kernel_size, strides=stride,
                     activation=activation)
                # SpectralNormalization(Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride,
                # activation=activation, padding="same"))
            ]
        else:
            layers = [
                conv(filters=out_channels, kernel_size=kernel_size, strides=stride,
                     activation=activation, padding=pad)
            ]
        self.layers = Sequential(layers)

    def call(self, x):
        return self.layers(x)


class ConvNormLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride, activation=True, frn=False, transpose=False, pad="refl"):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        if frn:
            layers = [
                ConvLayer(out_channels, kernel_size,
                          stride, transpose=transpose, pad=pad),
                FRN(),
            ]
            if activation:
                layers.append(TLU())
                # layers.append(ReLU())
        else:
            layers = [
                ConvLayer(out_channels, kernel_size,
                          stride, transpose=transpose, pad=pad),
                InstanceNormalization(axis=3, center=True, scale=True),
                # tf.keras.layers.BatchNormalization(axis=3, center=True, scale=True),
            ]
            if activation:
                layers.append(ReLU())

        self.layers = Sequential(layers)

    def call(self, x):
        return self.layers(x)


class ResLayer(Layer):

    def __init__(self, out_channels, kernel_size, reduction, frn=False):
        super().__init__()
        self.branch = Sequential([
            ReflectionPad2d(kernel_size // 2),
            Conv2D(filters=out_channels, kernel_size=kernel_size, activation="relu"),
            ReflectionPad2d(kernel_size // 2),
            Conv2D(filters=out_channels, kernel_size=kernel_size)
        ])
        self.ca_module = CALayer(out_channels, out_channels // reduction)
        if frn:
            self.activation = TLU()
        else:
            self.activation = ReLU()

    def call(self, x):
        y = self.branch(x)
        y = self.activation(y)
        # return y
        x = x + self.ca_module(y)
        return x


class CALayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = Sequential([
            Conv2D(filters=channel // reduction,
                   kernel_size=1, activation="relu", padding="valid"),
            Conv2D(filters=channel, kernel_size=1,
                   activation="sigmoid", padding="valid"),
        ])

        # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        # nn.Sigmoid()

    def call(self, x):
        y = self.avg_pool(x)
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        y = self.conv_du(y)
        return x * y


class ConvNoTanhLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = Sequential([
            ConvLayer(out_channels, kernel_size, stride),
        ])

    def call(self, x):
        return self.layers(x)


class ConvTanhLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = Sequential([
            ConvLayer(out_channels, kernel_size, stride, activation="tanh"),
        ])

    def call(self, x):
        return self.layers(x)


class ResModel(Model):
    def __init__(self, cfg):
        RM = cfg["model"]["ResModel"]
        N, d, reduction, frn = RM["N"], RM["d"], RM["reduction"], RM["frn"]
        super().__init__()

        self.layers_middle = [ConvLayer(d, 3, 1, pad="refl")]
        for i in range(N):
            self.layers_middle.append(
                ResLayer(d, 3, reduction, frn=frn))

        self.conv = ConvNoTanhLayer(3, 3, 1)
        # self.conv = ConvTanhLayer(3, 3, 1)
        # self.ca_module_resnets = CALayer(filter_counts[3], reduction)
        # self.ca_module_preprocess = CALayer(filter_counts[0], reduction)
        # self.ca_module_global = CALayer(filter_counts[0] * 2, reduction)

    def call(self, x, training=False):
        remember = x
        # remember = []
        # for layer in self.layers_preprocess:
        # x = layer(x)
        for layer in self.layers_middle:
            x = layer(x)
        # x += self.ca_module_resnets(y)

        return self.conv(x) + remember
