import torch
import timm
from custom_models.mobilenetv2 import MobileNetV2Q
import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras


def load_model(filename, modelname, jit=False, quantized=False,
               example_input=None, make_jit=True, num_classes=2):
    if jit:
        jit_model = torch.jit.load(filename)
        return jit_model

    state = torch.load(filename)

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
            "fbgemm")
        torch.quantization.prepare_qat(torch_model.eval(), inplace=True)

    torch_model.load_state_dict(state)
    if quantized:
        torch.quantization.convert(torch_model, inplace=True)

    if make_jit:
        torch_model.eval()
        jit_model = torch.jit.trace(torch_model, example_input)
        return jit_model

    return torch_model


def quantize(model):
    model.to("cpu")
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig(
        "fbgemm")
    torch.quantization.prepare_qat(model.eval(), inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.eval()
    return model


def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    onnx_model = onnx.load(onnx_model_path)

    input_names = ['image_array']
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            change_ordering=True, verbose=False)

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        tf.saved_model.simple_save(
            sess,
            str(saved_model_dir.joinpath('1')),
            inputs={'image_array': k_model.input},
            outputs=dict((output.name, tensor) for output, tensor in zip(
                onnx_model.graph.output, k_model.outputs))
        )


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False):
    saved_model_dir = str(Path(saved_model_dir).joinpath('1'))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model
