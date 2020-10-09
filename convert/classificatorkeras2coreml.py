import tensorflow as tf
import coremltools as ct
import tensorflow_model_optimization as tfmot
import argparse
from utils import url2image, default_transformation
from tqdm import tqdm
import numpy as np
from pprint import pprint
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


trans = default_transformation(framework="keras")


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--keras_file", type=str, help="Path to torch file")
    args.add_argument("--save_path", type=str,
                      help="Path for saving tflite file")
    args.add_argument("--labels_file", type=str, help="Path to labels file")
    args.add_argument("--quantized", default=False, action="store_true",
                      help="Do we need model quantization before evaluation?")
    args.add_argument("--test_path", type=str, default=None,
                      help="Path to .txt file with urls to compare models")
    
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
    class_labels = [x for x in class_labels if x]
    print(f"class labels: {class_labels}")
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
    
    with open(args.test_path, "r") as f:
        urls = f.read().split("\n")
    urls = [x for x in urls if x]

    d = []

    for url in tqdm(urls):
        image = url2image(url, (input_shape[1], input_shape[2]))
        transformed_image = trans(image=np.array(image))["image"]
        transformed_image = np.expand_dims(transformed_image, axis=0)
        keras_pred = keras_model.predict(transformed_image)
        print(f"keras pred: {keras_pred}")
        ct_pred = ct_model.predict({"input_1": image}) #may be input_1
        d.append({
            "url": url,
            "ct": ct_pred,
            "keras": keras_pred,
            "image": image
        })
    d = sorted(d, key=lambda x: x["keras"][0][1])
    pprint(d)
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
    d = sorted(d, key=lambda x: x['ct']['Identity']['beauty'])
    with PdfPages("beauty_coreml.pdf") as pdf:        
       for el in tqdm(d):
           try:
                img = el["image"]
                score = el["ct"]['Identity']['beauty']
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



    ct_model.save(args.save_path)
