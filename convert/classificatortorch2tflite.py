import torch
import argparse
import coremltools as ct
import timm
from utils import (
    load_model,
    pytorch2savedmodel,
    savedmodel2tflite)
import torch.nn as nn
import io


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--torch_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving tflite file")
    args.add_argument("--model", type=str,
                      help="Model class from timm", default="")
    args.add_argument("--num_classes", type=int,
                      help="Number classes from timm", default=2)
    args.add_argument("--jit", default=False, action="store_true",
                      help="Is model stored in jit format?")
    args.add_argument("--quantized", default=False, action="store_true",
                      help="Do we need model quantization before evaluation?")
    args.add_argument("--size_x", type=int, help="Horizontal size of image")
    args.add_argument("--size_y", type=int, help="Vertical size of image")
    args.add_argument("--model_dir", type=str,
                      help="Choose directory for model saving")
    args.add_argument("--qconfig", type=str,
                      help="fbgemm or qnnpack")

    args = args.parse_args()
    example_input = torch.rand(1, 3, args.size_x, args.size_y)
    model = load_model(args.torch_file, args.model, jit=args.jit,
                       quantized=args.quantized, example_input=example_input,
                       make_jit=True, num_classes=args.num_classes)
    for m in model.modules():
        m.training = False

    model.eval()
    onnx_model_path = "model.onnx"
    input_names = ["image_array"]
    output_names = ["classification"]
    example_output = model(example_input)
    torch.onnx.export(model, example_input, onnx_model_path, example_outputs=example_output,
                      input_names=input_names, output_names=output_names,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      opset_version=12)
    print("onnx2tflite")
    pytorch2savedmodel(onnx_model_path, args.model_dir)
    tflite_model = savedmodel2tflite(
        args.model_dir, args.save_path, quantize=args.quantized,
        input_names=input_names, output_names=output_names)
