import tensorflow as tf
import os
import functools
import json
import numpy as np

from tqdm import tqdm
from tools import reader
from models import semi_alexnet_v1

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
    images_root = os.path.join(dataset_path, IMAGES_DIR, subset)
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


def dataset_from_file_iterator(iter, cat2id, batch_size):
    file_dataset = tf.data.Dataset.from_generator(
        iter,
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

    dataset = dataset.take(1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


BATCH_SIZE = 32


if __name__ == "__main__":
    with open("cat2id.json") as f:
        cat2id = json.load(f)
    print len(cat2id), "categories"

    train_dataset = dataset_from_file_iterator(
        functools.partial(image_annotation_iterator, "./ILSVRC", subset="train"),
        cat2id,
        BATCH_SIZE
    )
    valid_dataset = dataset_from_file_iterator(
        functools.partial(image_annotation_iterator, "./ILSVRC", subset="val"),
        cat2id,
        BATCH_SIZE
    )
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )

    train_initializer_op = iterator.make_initializer(train_dataset)
    valid_initializer_op = iterator.make_initializer(valid_dataset)

    img_batch, label_batch = iterator.get_next()

    logits = semi_alexnet_v1.semi_alexnet_v1(img_batch, len(cat2id) + 1)
    loss = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=label_batch)

    labels = tf.argmax(label_batch, axis=1)
    predictions = tf.argmax(logits, axis=1)

    correct_predictions = tf.reduce_sum(tf.to_float(tf.equal(labels, predictions)))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver()

    EPOCHS = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epochs in range(EPOCHS):
            counter = tqdm()
            sess.run(train_initializer_op)
            total = 0.
            correct = 0.
            try:
                while True:
                    opt, l, correct_batch = sess.run([optimizer, loss, correct_predictions])
                    total += BATCH_SIZE
                    correct += correct_batch
                    counter.set_postfix({
                        "loss": "{:.6}".format(l),
                        "accuracy": correct/total
                    })
                    counter.update(BATCH_SIZE)
            except tf.errors.OutOfRangeError:
                print "Finished training"

            counter = tqdm()
            sess.run(valid_initializer_op)
            total = 0.
            correct = 0.
            try:
                while True:
                    l, correct_batch = sess.run([loss, correct_predictions])
                    total += BATCH_SIZE
                    correct += correct_batch
                    counter.set_postfix({
                        "loss": "{:.6}".format(l),
                        "valid accuracy": correct/total
                    })
                    counter.update(BATCH_SIZE)
            except tf.errors.OutOfRangeError:
                print "Finished validation"

            saver.save(sess, "model/checkpoint_1")