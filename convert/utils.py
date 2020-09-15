import torch
import torch.nn as nn
import timm
from custom_models.mobilenetv2 import MobileNetV2Q
import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras
import subprocess


def load_model(filename, modelname, jit=False, quantized=False,
               example_input=None, make_jit=True, num_classes=2,
               qconfig="qnnpack", add_softmax=False):
    if jit:
        jit_model = torch.jit.load(filename)
        return jit_model

    if filename is not None:
        state = torch.load(filename, map_location="cpu")

    if modelname != "MobileNetV2Q":
        torch_model = timm.create_model(
            modelname, pretrained=False, num_classes=num_classes)
    else:
        torch_model = MobileNetV2Q(num_classes=num_classes)

    if quantized:
        torch_model.to("cpu")
        torch_model.eval()
        torch_model.fuse_model()
        torch_model.qconfig = torch.quantization.get_default_qat_qconfig(
            qconfig)
        torch.quantization.prepare_qat(torch_model.eval(), inplace=True)
        torch.quantization.convert(torch_model, inplace=True)

    if filename is not None:
        torch_model.load_state_dict(state)

    if add_softmax:
        if example_input is not None:
            print(torch_model(example_input))
        torch_model = nn.Sequential(torch_model, nn.Softmax())
        if example_input is not None:
            print("after adding softmax")
            print(torch_model(example_input))

    if make_jit:
        torch_model.eval()
        jit_model = torch.jit.trace(torch_model, example_input)
        return jit_model

    return torch_model


def quantize(model, qconfig):
    model.to("cpu")
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig(
        qconfig)
    torch.quantization.prepare_qat(model.eval(), inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.eval()
    return model


def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    subprocess.check_output(
        ['onnx-tf', 'convert', '-i', onnx_model_path, "-o",
         saved_model_dir + "/saved_model.pb"])


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False,
                      input_names=["image_array"], output_names=["classification"]):

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=saved_model_dir + "/saved_model.pb",
        input_arrays=input_names,
        output_arrays=output_names
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('model.tflite', 'wb') as f:
        f.write(tflite_model)
