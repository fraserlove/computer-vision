import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Import the TensorFlow Object Detection API visualisation tools.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def load_label_map(path):
    print(f'Loading label map - {path}')
    return label_map_util.create_category_index_from_labelmap(path)

def load_model(path):
    print(f'Loading model - {path}')
    return hub.load(path) 

def run_model(model, image):
    # Add an axis as model expects a tensor representing a batch of images.
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image), 0)
    # TensorFlow Object Detection API expects detections to be numpy arrays.
    return {key:value.numpy() for key,value in model(image_tensor).items()}

def show_detections(image, results, label_map, threshold = 0.3):
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image,
          results['detection_boxes'][0],
          (results['detection_classes'][0]).astype(int),
          results['detection_scores'][0],
          label_map,
          use_normalized_coordinates=True,
          min_score_thresh=threshold)
    cv2.imshow('Object Detection', annotated_image)
    

if __name__ == '__main__':

    IN_VIDEO_PATH = 'input_video.mp4'
    OUT_VIDEO_PATH = 'output_video.mp4'
    LABEL_MAP_PATH = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

    # EfficientDet D7 1536x1536.
    MODEL_PATH = 'https://tfhub.dev/tensorflow/efficientdet/d7/1'

    labels = load_label_map(LABEL_MAP_PATH)
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(IN_VIDEO_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUT_VIDEO_PATH, codec, fps, (frame_width, frame_height))
	
    while True:
        try:
            ret, image = cap.read()

            results = run_model(model, image)
            annotated_image = image.copy()
            show_detections(annotated_image, results, labels, threshold=0.45)
            out.write(annotated_image)

            # Press 'q' key to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print('Finished Processing'); break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

