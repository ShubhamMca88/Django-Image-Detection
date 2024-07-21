import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the pre-trained model using the recommended function
model = tf.compat.v2.saved_model.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    return image

def remove_background(image_path):
    image = load_image(image_path)
    input_image = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_image = tf.image.resize(input_image, (640, 640))
    input_image = tf.expand_dims(input_image, axis=0)
    
    detections = model(input_image)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    
    # Filter detections with a threshold score
    threshold = 0.5
    mask = detection_scores > threshold
    detection_boxes = detection_boxes[mask]
    
    # Example process: Create a mask for the detected objects
    image_height, image_width, _ = image.shape
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    for box in detection_boxes:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)
        mask[ymin:ymax, xmin:xmax] = 1
    
    # Apply mask to the image to remove the background
    result = image * np.expand_dims(mask, axis=-1)
    result_image = Image.fromarray(result)
    
    return result_image

