#!/usr/bin/env python
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def input_parser(file_name, labels):

    image_data = tf.read_file(file_name)
    image_parsed = tf.image.decode_jpeg(image_data, channels=3)
    print image_parsed.shape
    image_parsed = tf.image.resize_image_with_crop_or_pad(image_parsed, 482, 415)
    image_parsed = tf.cast(image_parsed, dtype=tf.float16)
    image_parsed = tf.image.per_image_standardization(image_parsed)
    return image_parsed, labels


BATCH_SIZE = 10

def bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=val))

def int_feature(val):
    try:
       if not isinstance(val, (tuple, list)):
           val = [val]
       return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    except TypeError:
        print val
        raise

def create_example(image, label):
    return tf.train.Example(
        features=tf.train.Features(feature={
            "image": bytes_feature(image.tobytes()),
            "height": int_feature(image.shape[0]),
            "width": int_feature(image.shape[1]),
            "label": bytes_feature(label.tobytes())
        })
    )

if __name__ == "__main__":
    FILE = "one_biggest_3000.csv"
    data = pd.read_csv(FILE, index_col=0)
    dataset = tf.data.Dataset.from_tensor_slices((data.Image.as_matrix(), pd.get_dummies(data.class_id)))
    dataset = dataset.map(input_parser)
    dataset = dataset.batch(BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    img_label = iterator.get_next()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.python_io.TFRecordWriter("converted_data") as writer:
            pbar = tqdm(total=3000)
            while True:
                try:
                    images = sess.run(img_label)

                    for img, label in zip(images[0], images[1]):
                        example = create_example(img, label)
                        writer.write(example.SerializeToString()) 
                except tf.errors.OutOfRangeError:
                    print "finished"
                    break
                pbar.update(BATCH_SIZE)
