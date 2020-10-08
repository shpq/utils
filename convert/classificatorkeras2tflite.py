import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.lite import TFLiteConverter
from tensorflow.keras.models import load_model
from utils import url2image, default_transformation
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import argparse
from pprint import pprint
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm

trans = default_transformation(framework="keras")

if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--keras_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving tflite file")
    args.add_argument("--quantized", default=False, action="store_true",
                      help="Do we need model quantization before evaluation?")
    args.add_argument("--test_path", type=str, default=None,
                      help="Path to .txt file with urls to compare models")

    args = args.parse_args()

    with tfmot.quantization.keras.quantize_scope():
        keras_model = keras.models.load_model(args.keras_file)

    keras_model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if args.test_path:
        with open(args.test_path, "r") as f:
            urls = f.read().split("\n")

    def representative_dataset():
        return [[np.expand_dims(trans(image=np.array(url2image(
            url, (keras_model.input.shape[1], keras_model.input.shape[2]))))["image"], axis=0)] for url in urls]

    if args.quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter._experimental_new_quantizer = True

    tflite_model = converter.convert()

    with open(args.save_path, "wb") as f:
        f.write(tflite_model)

    if args.test_path:

        tflite_model = tf.lite.Interpreter(model_path=args.save_path)
        tflite_model.allocate_tensors()

        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        input_shape = input_details[0]["shape"]
        with open(args.test_path, "r") as f:
            urls = f.read().split("\n")
        d = []

        for url in tqdm(urls):
            image = url2image(url, (input_shape[1], input_shape[2]))
            transformed_image = trans(image=np.array(image))["image"]
            transformed_image = np.expand_dims(transformed_image, axis=0)
            keras_pred = keras_model.predict(transformed_image)
            tflite_model.set_tensor(
                input_details[0]["index"], transformed_image)
            tflite_model.invoke()
            tflite_pred = tflite_model.get_tensor(output_details[0]["index"])

            d.append({
                "url": url,
                "keras": keras_pred,
                "tflite": tflite_pred,
                "image": image
            })

        d = sorted(d, key=lambda x: x["tflite"][0][1])
        with PdfPages("beauty_tflite.pdf") as pdf:
            for el in tqdm(d):
                try:
                    img = el["image"]
                    score = el["tflite"][0][1]
                    url = el["url"]
                    plt.figure(figsize=(20, 20))
                    plt.title(f"{score}\n {url}")
                    plt.imshow(img)

                    pdf.savefig()
                    plt.clf()
                    plt.close()

                except Exception as e:
                    print(e)
                    continue

        d = sorted(d, key=lambda x: x["keras"][0][1])

        with PdfPages("beauty_keras.pdf") as pdf:
            for el in tqdm(d):
                try:
                    img = el["image"]
                    score = el["keras"][0][1]
                    url = el["url"]
                    plt.figure(figsize=(20, 20))
                    plt.title(f"{score}\n {url}")
                    plt.imshow(img)

                    pdf.savefig()
                    plt.clf()
                    plt.close()

                except Exception as e:
                    print(e)
                    continue
