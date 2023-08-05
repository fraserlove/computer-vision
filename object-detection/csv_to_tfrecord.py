import tensorflow as tf
import pandas as pd
import os, io, collections, sys, time
from PIL import Image
from object_detection.utils import dataset_util


flags = tf.compat.v1.flags
flags.DEFINE_string('input_path', '', 'Path to input CSV')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

label_map_dict =  {
    'face' : 1,
    }

def create_tf_example(group, path):
    
    filename = group.filename.encode()
    image_format = 'jpg'.encode() 

    with tf.io.gfile.GFile(f'{path}/{group.filename}', 'rb') as fid:
        encoded_image = fid.read()

    image = Image.open(io.BytesIO(encoded_image))
    width, height = image.size

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for i, row in group.example.iterrows():
        xmins.append(row['x0'] / width)
        xmaxs.append(row['x1'] / width)
        ymins.append(row['y0'] / height)
        ymaxs.append(row['y1'] / height)

        key = row['label'] if 'label' in row else next(iter(label_map_dict))
        classes_text.append(key.encode())
        classes.append(label_map_dict[key])


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    with tf.io.TFRecordWriter(FLAGS.output_path) as tf_writer:
        examples_df = pd.read_csv(FLAGS.input_path)
        
        # Group rows in the CSV by filename
        data = collections.namedtuple('data', ['filename', 'example'])
        groups_df = examples_df.groupby('image_name')
        groups = [data(filename, groups_df.get_group(group_name)) for filename, group_name in zip(groups_df.groups.keys(), groups_df.groups)]

        path = '/'.join(FLAGS.input_path.split('/')[:-1])
        
        print(f'CSV to TFRecord Convertor: Loading {len(examples_df)} examples')
        for i, group in loading_bar(groups, prefix='Progress:', suffix='Complete', length=50):
                tf_example = create_tf_example(group, path)
                tf_writer.write(tf_example.SerializeToString())


def loading_bar(iterable, prefix='', suffix='', length=50, fill='█'):
    start_time = time.time()

    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f'{m:02d}:{s:02d}'

    def update_bar(progress):
        filled_length = int(length * progress)
        bar = fill * filled_length + '-' * (length - filled_length)
        elapsed_time = time.time() - start_time
        time_str = format_time(elapsed_time)
        sys.stdout.write(f'\r{prefix} |{bar}| {progress * 100:.1f}% {suffix} - Elapsed: {time_str}')
        sys.stdout.flush()

    for i, item in enumerate(iterable, 1):
        yield i, item
        update_bar(i / len(iterable))

    sys.stdout.write('\n')


if __name__ == '__main__':
    tf.compat.v1.app.run()