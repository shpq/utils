import torch
from utils import (
    load_model, quantize)
import argparse
import timm


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--torch_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving mlmodel file")
    args.add_argument("--model", type=str,
                      help="Model class from timm", default="")
    args.add_argument("--num_classes", type=int,
                      help="Number classes from timm", default=2)
    args.add_argument("--jit", default=False, action="store_true",
                      help="Is model stored in jit format?")
    args.add_argument("--size_x", type=int, help="Horizontal size of image")
    args.add_argument("--size_y", type=int, help="Vertical size of image")

    args = args.parse_args()

    example_input = torch.rand(1, 3, args.size_x, args.size_y)

    model = load_model(args.torch_file, args.model, jit=False, quantized=False,
                       example_input=example_input, make_jit=False,
                       num_classes=args.num_classes)
    
    quantized_model = quantize(model)
    torch.save(quantized_model.state_dict(), args.save_path)
