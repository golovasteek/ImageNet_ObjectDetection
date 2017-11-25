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


if __name__ == "__main__":
    FILE = "one_biggest_3000.csv"
    data = pd.read_csv(FILE, index_col=0)
    dataset = tf.data.Dataset.from_tensor_slices((data.Image.as_matrix(), pd.get_dummies(data.class_id)))
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    img_label = input_parser(*next_elem)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.python_io.TFRecordWriter("converted_data") as writer:
            pbar = tqdm(total=3000)
            while True:
                try:
                    img = sess.run(img_label)
                    example = tf.train.Example(
                        features=tf.train.Features(feature={
                            "image": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=img[0].tostring()))
                        })
                    )
                    writer.write(example.SerializeToString())
                except tf.errors.OutOfRangeError:
                    print "finished"
                pbar.update(1)