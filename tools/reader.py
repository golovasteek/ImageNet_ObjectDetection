import os
import bs4
import requests
import matplotlib.pyplot as plt
import math

CACHE={}

TRAIN_DATA_PATH="ILSVRC/Data/DET/train/"
TRAIN_ANNOTATION_PATH="ILSVRC/Annotations/DET/train/"


class Object(object):
    def __init__(self, x1, y1, x2, y2, cls):
        self.xmin = x1
        self.ymin = y1
        self.xmax = x2
        self.ymax = y2
        self.cls = cls

    @property
    def area(self):
        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return "{}: ({}, {}) ({}, {})".format(
            self.cls,
            self.xmin, self.ymin,
            self.xmax, self.ymax
        )


class Annotation(object):
    def __init__(self, path):
        self.path = path

        self.objects = []

    def main_object(self):
        if not self.objects:
            return Object(0, 0, 1000000, 100000, "<backgroud>")
        result = self.objects[0]
        for obj in self.objects:
            if obj.area > result.area:
                result = obj
        return result

    def __repr__(self):
        return "\n\t".join(
            [self.path] +
            [obj.__repr__() for obj in self.objects])


def read_annotations():
    for dir_path, _, file_names in os.walk(TRAIN_ANNOTATION_PATH):
        for fname in file_names:
            path = os.path.join(dir_path, fname)
            with open(path) as f:
                parsed = bs4.BeautifulSoup(f, 'lxml')
                relpath = os.path.relpath(path, TRAIN_ANNOTATION_PATH)
                img_path = os.path.splitext(os.path.join(TRAIN_DATA_PATH, relpath))[0] + '.JPEG'
                if not os.path.exists(img_path):
                    raise RuntimeError("Image {} not found on disk".format(img_path))
                result = Annotation(img_path)
                result.objects = [
                    Object(
                        int(obj.xmin.text), int(obj.ymin.text),
                        int(obj.xmax.text), int(obj.ymax.text),
                        obj.find("name").text.decode()
                    )
                    for obj in parsed.find_all("object")
                ]
                yield result


def get_desc(wnidx):
    if wnidx == "<background>":
        return "<backgroud>"
    if wnidx not in CACHE:
        url = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}".format(wnidx)
        try:
            CACHE[wnidx] = " ".join(requests.get(url).text.split())
        except:
            print "Failed to load url:", url
            raise
    return CACHE[wnidx]


def list_file_class():
    for annotation in read_annotations():
        cls = annotation.main_object().cls
        yield annotation.path, cls, get_desc(cls)


def preview_image(annotation):
    plt.imshow(plt.imread(annotation.path))
    ax = plt.axes()
    for obj in [annotation.main_object()]:
        ax.add_patch(patches.Rectangle(
            (obj.xmin, obj.ymin),
            obj.xmax - obj.xmin, obj.ymax - obj.ymin,
            fill=False, color=plt.cm.jet(hash(obj.cls) % 256), label=get_desc(obj.cls)))

    ax.legend()