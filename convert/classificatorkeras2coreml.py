import tensorflow as tf
import coremltools as ct
import tensorflow_model_optimization as tfmot
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--keras_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving tflite file")
    args.add_argument("--labels_file", type=str, help="Path to labels file")
    args.add_argument("--quantized", default=False, action="store_true",
                      help="Do we need model quantization before evaluation?")
    args = args.parse_args()
    if args.quantized:
        with tfmot.quantization.keras.quantize_scope():
            keras_model = tf.keras.models.load_model(args.keras_file)
    else:
        keras_model = tf.keras.models.load_model(args.keras_file)
    input_shape = list(keras_model.input.shape)
    input_shape[0] = 1
    input_shape = tuple(input_shape)

    with open(args.labels_file, "r") as f:
        class_labels = f.read()
        class_labels = class_labels.split("\n")

    scale = 1.0 / 255.0 / 0.226
    bias = [-0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225]

    image_input = ct.ImageType(
        name="input_1", shape=input_shape,
        bias=bias, scale=scale)
    classifier_config = ct.ClassifierConfig(class_labels)

    ct_model = ct.convert(keras_model, inputs=[image_input],
                          image_input_names='image_input',
                          classifier_config=classifier_config,
                          source="tensorflow")
    ct_model.save(args.save_path)
