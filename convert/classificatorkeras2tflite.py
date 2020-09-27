import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.lite import TFLiteConverter
from tensorflow.keras.models import load_model
from utils import url2image, default_transformation

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
    converter = TFLiteConverter.from_keras_model_file(args.keras_file)
    if args.quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(args.save_path, "wb") as f:
        f.write(tflite_model)

    if args.test_path:
        keras_model = load_model(args.keras_file)
        tflite_model = tf.lite.Interpreter(model_path=args.save_path)
        tflite_model.allocate_tensors()
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        input_shape = input_details[0]["shape"]
        with open(args.test_path, "r") as f:
            urls = f.read().split("\n")
        trans = default_transformation(framework="keras")
        d = []
        for url in urls:
            image = url2image(url, (input_shape[1], input_shape[2]))
            transformed_image = trans(np.array(image))
            # maybe wrapped?
            keras_pred = keras_model.predict(
                np.expand_dims(transformed_image, axis=0))
            tflite_model.set_tensor(input_details[0]["index"], transformed_image)
            tflite_model.invoke()
            tflite_pred = tflite_model.get_tensor(output_details[0]["index"])

            d.append({
                "url": url,
                "keras": keras_pred,
                "tflite": tflite_predm
            })
        pprint(d)
