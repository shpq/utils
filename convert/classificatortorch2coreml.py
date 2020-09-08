import torch
import argparse
import coremltools as ct
import timm
from utils import load_model


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--torch_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving mlmodel file")
    args.add_argument("--labels_file", type=str, help="Path to labels file")
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

    args = args.parse_args()

    example_input = torch.rand(1, 3, args.size_x, args.size_y)

    jit_model = load_model(args.torch_file, args.model, jit=args.jit,
                           quantized=args.quantized, example_input=example_input,
                           make_jit=True, num_classes=args.num_classes)

    with open(args.labels_file, "r") as f:
        class_labels = f.read()
        class_labels = class_labels.split("\n")

    # AttributeError: 'tuple' object has no attribute 'detach' ??
    ct_model = ct.convert(jit_model, inputs=[ct.ImageType(
        name="input", shape=example_input.shape)],
        classifier_config=ct.ClassifierConfig(class_labels))

    ct_model.save(args.save_path)
