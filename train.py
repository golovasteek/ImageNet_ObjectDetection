import tensorflow as tf
import os


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


if __name__ == "__main__":
    for i, files in zip(range(10), image_annotation_iterator("./ILSVRC")):
        print i, files

    print "Success"