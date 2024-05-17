import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the pre-trained model
model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    return image

def detect_objects(image_path):
    image = load_image(image_path)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)[tf.newaxis, ...]
    result = model(image_tensor)
    return result
