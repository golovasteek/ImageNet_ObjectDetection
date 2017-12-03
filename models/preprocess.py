import tensorflow as tf


def input_parser(file_name, labels):
    with tf.device("/cpu:0"):
        image_data = tf.read_file(file_name)
        image_parsed = tf.image.decode_jpeg(image_data, channels=3)
        print image_parsed.shape
        image_parsed = tf.image.resize_image_with_crop_or_pad(image_parsed, 482, 415)
        image_parsed = tf.cast(image_parsed, dtype=tf.float16)
        image_parsed = tf.image.per_image_standardization(image_parsed)
        return image_parsed, labels