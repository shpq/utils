import torch
import argparse
import coremltools as ct
import timm
from utils import load_model, url2image, default_transformation
from add_some_ops.torch import constant_pad_nd
import numpy as np
from pprint import pprint


description = "Example: python3 classificatortorch2coreml.py " +\
    "--save_path models/beauty.mlmodel --labels_file labels/beauty_labels.txt" +\
    " --model MobileNetV2Q --num_classes 2 --size_x 600 --size_y 600 --add_softmax"

if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                   description=description)
    args.add_argument("--torch_file", type=str, help="Path to torch file",
                      default=None)
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
    args.add_argument("--add_softmax", default=False, action="store_true",
                      help="Add softmax layer to the end of NN?")
    args.add_argument("--size_x", type=int, help="Horizontal size of image")
    args.add_argument("--size_y", type=int, help="Vertical size of image")
    args.add_argument("--qconfig", type=str, default=None,
                      help="fbgemm or qnnpack")
    args.add_argument("--test_path", type=str, default=None,
                      help="Path to .txt file with urls to compare models")

    args = args.parse_args()

    example_input = torch.rand(1, 3, args.size_x, args.size_y)

    jit_model = load_model(args.torch_file, args.model, jit=args.jit,
                           quantized=args.quantized, example_input=example_input,
                           make_jit=True, num_classes=args.num_classes, qconfig=args.qconfig,
                           add_softmax=args.add_softmax)

    with open(args.labels_file, "r") as f:
        class_labels = f.read()
        class_labels = class_labels.split("\n")

    scale = 1.0 / 255.0 / 0.226
    bias = [-0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225]

    image_input = ct.ImageType(
        name="image_input", shape=example_input.shape,
        bias=bias, scale=scale)


    jit_model.eval()
    ct_model = ct.convert(jit_model, inputs=[image_input],
                          classifier_config=ct.ClassifierConfig(class_labels))
    if args.test_path:
        with open(args.test_path, "r") as f:
            urls = f.read().split("\n")
        trans = default_transformation()
        d = []
        for url in urls:
            image = url2image(url, (args.size_x, args.size_y))
            transformed_image = trans(np.array(image))
            pred_jit = jit_model(torch.unsqueeze(transformed_image, 0))
            d.append({
                "url": url,
                "jit": pred_jit,
            })
        pprint(d)

    ct_model.save(args.save_path)
