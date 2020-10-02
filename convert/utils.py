import numpy as np
import torch
import torch.nn as nn
import timm
from custom_models.mobilenetv2 import MobileNetV2Q
import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
import subprocess
import requests
from PIL import Image
from torchvision import transforms
import onnx
from onnx_tf.backend import prepare
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(filename, modelname, jit=False, quantized=False,
               example_input=None, make_jit=True, num_classes=2,
               qconfig="qnnpack", add_softmax=False):
    if jit:
        jit_model = torch.jit.load(filename)
        return jit_model

    if filename is not None:
        state = torch.load(filename, map_location="cpu")

    if modelname.startswith("hub_"):
        torch_model = torch.hub.load(
            'pytorch/vision:v0.6.0', modelname[4:], pretrained=False)
        torch_model.classifier[1] = nn.Linear(
            torch_model.classifier[1].in_features, num_classes)
    elif modelname != "MobileNetV2Q":
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


def savedmodel2tflite(onnx_model_path, saved_model_dir, tflite_model_path, quantize=False,
                      input_names=["image_array"], output_names=["classification"]):
    model_onnx = onnx.load(onnx_model_path)
    tf_rep = prepare(model_onnx)
    print('inputs:', tf_rep.inputs)
    print('outputs:', tf_rep.outputs)
    tf_rep.export_graph(saved_model_dir + "/saved_model.pb")
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=saved_model_dir + "/saved_model.pb",
        input_arrays=tf_rep.inputs,
        output_arrays=tf_rep.outputs
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


def default_transformation(framework="torch"):
    normalize = A.Normalize()
    if framework == "torch":
        return A.Compose([
            normalize,
            ToTensorV2()])
    elif framework in ["tflite", "keras"]:
        return A.Compose([
            normalize])
    else:
        raise NotImplementedError


def url2image(url, size):
    return Image.open(requests.get(url, stream=True).raw).resize(size)
