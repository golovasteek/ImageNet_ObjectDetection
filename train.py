import tensorflow as tf
import os
import functools
import json
from tools import reader
import numpy as np


ANNOTATION_DIR = os.path.join("Annotations", "DET")
IMAGES_DIR = os.path.join("Data", "DET")
IMAGES_EXT = "JPEG"


def image_annotation_iterator(dataset_path, subset="train"):
    """
    Yields tuples of image filename and corresponding annotation.

    :param dataset_path: Path to the root of uncompressed ImageNet dataset
    :param subset: one of 'train', 'val', 'test'
    :return: iterator
    """

    annotations_root = os.path.join(dataset_path, ANNOTATION_DIR, subset)
    print annotations_root
    images_root = os.path.join(dataset_path, IMAGES_DIR, subset)
    print images_root
    for dir_path, _, file_names in os.walk(annotations_root):
        for annotation_file in file_names:
            path = os.path.join(dir_path, annotation_file)
            relpath = os.path.relpath(path, annotations_root)
            img_path = os.path.join(
                images_root,
                os.path.splitext(relpath)[0] + '.' + IMAGES_EXT
            )
            assert os.path.isfile(img_path), RuntimeError("File {} doesn't exist".format(img_path))
            yield img_path, path


def image_parser(file_name):
    image_data = tf.read_file(file_name)
    image_parsed = tf.image.decode_jpeg(image_data, channels=3)
    image_parsed = tf.image.resize_image_with_crop_or_pad(image_parsed, 482, 415)
    image_parsed = tf.cast(image_parsed, dtype=tf.float16)
    image_parsed = tf.image.per_image_standardization(image_parsed)
    return image_parsed


if __name__ == "__main__":
    with open("cat2id.json") as f:
        cat2id = json.load(f)
    print len(cat2id), "categories"
    file_dataset = tf.data.Dataset.from_generator(
        functools.partial(image_annotation_iterator, "./ILSVRC"),
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
    )

    def ann_file2one_hot(ann_file):
        annotation = reader.Annotation("unused", ann_file)
        category = annotation.main_object().cls
        result = np.zeros(len(cat2id) + 1)
        result[cat2id.get(category, len(cat2id))] = 1
        return result

    dataset = file_dataset.map(
        lambda img_file_tensor, ann_file_tensor:
            (
                image_parser(img_file_tensor),
                tf.py_func(ann_file2one_hot, [ann_file_tensor], tf.float64)
            )
    )

    BATCH_SIZE = 16
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()
    print type(next_elem[0])
    with tf.Session() as sess:
        for i in range(3):
            element = sess.run(next_elem)
            print i, element[0].shape, element[1].shape
    print "Success"