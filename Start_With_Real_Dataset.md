О чем эта статья
================
Глубокие нейронные сети сейчас модная тема.
В Сети много тюториалов и видеолекций, и других материалов обсуждающих 
основные принципы, архитектуру, стратегии обучения и т.д.
При этом большинство руководств обходят стороной вопросы загрузки
данных для обучения и сохранения модели, для последующиего использования.
Я хочу рассказать о том, как загружать данные при треннировке нейронных
сетей с использованием [Tensorflow][1].
В качестве примера задачи будет использован датасет [ImageNet][2]
опубликованный недавно в качестве [соревнования по детектированию объектов на Kaggle][3]
Мы будем обучать сеть детектированию одного объекта, того у
которого самый большой ограничивающий прямоугольник.

Подготовительные шаги
=====================
Ниже предполагается, что у вас
  * установлен [Python][python_org], в примерах используется Python 2.7,
  но не должно быть сложностей с их портированием на Python 3.* 
  * становлена библитека [Tensorflow и Python-интерфейс к ней][install_tensorflow]
  * скачан и распакован [набор данных][download_dataset] из соревнования на Kaggle 

Сразу импортируем нужные библиотеки:

```python
import os
import tensorflow as tf
# TODO: добавить остальные библиотеки
```

Препроцессинг данных
====================
Для загрузки данных, мы будем использовать механизмы, предоставляемые
[модулем для работы с датасетами][tf_dataset] в Tensorflow.
Для треннировки и валидации нам потребуется датасет в котором
одновременно и изображения и их описания. Но в скачаном датасете 
файлы с изображениями и аннотациями аккуратно разложены по разным папочкам.
Поэтму мы сделаем итератор который итерируется по соотвествующим парам.
```python
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
            assert os.path.isfile(img_path), \
                RuntimeError("File {} doesn't exist".format(img_path))
            yield img_path, path
```

Из этого можно уже сделать датасет и запустить "процессинг на графе",
например извлекать имена файлов из датасета.
Создаем датасет: 
```python
files_dataset = tf.data.Dataset.from_generator(
    functools.partial(image_annotation_iterator, "./ILSVRC"),
    output_types=(tf.string, tf.string),
    output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
)
```

Для извлечения данных из датасета нам нужен итератор
`make_one_shot_iterator` создаст итератор, который проходит по 
данным один раз. `Iterator.get_next()` создает тензор в вкоторый загружаются
данне из итератора.
```python
iterator = files_dataset.make_one_shot_iterator()
next_elem = iterator.get_next()
```
Теперь можно создать сессию и ее "вычислить значения" тензора:
```python
with tf.Session() as sess:
    for i in range(10):
        element = sess.run(next_elem)
        print i, element
```
Но нам для использования в нейронных сетях нужны не имена файлов, а изображения
в виде "трехслойных" матриц одинаковой формы и категории этих изображений в виде
["one hot"-вектора][one_hot_rus_wiki] 

### Кодируем категории изображений
Разбор файлов аннотаций, не очень инетересен сам по себе. Я использовал для 
этого пакет [BeautifulSoup][beautiful_soup]. Вспомогательный класс `Annotation`
умеет инициализироваться из пути к файлу и хранить список объектов.
Для начала нам надо собрать список категорий, чтобы знать размер вектора для
кодирования `cat_max`. А так же сделать отображение строковых
категорий в номер из `[0..cat_max]`. Создание таких отображений тоже не очень
интрерсно, дальше будем считать что словари `cat2id` и `id2cat` содержат
прямое и обратное отображение описнное выше.
Функция преобразования имени файла в закодированный векто категорий.
Видно что добавляется еще одна категория, для фона: на некоторых изображений
не отмечен ни один объект.
```python
def ann_file2one_hot(ann_file):
    annotation = reader.Annotation("unused", ann_file)
    category = annotation.main_object().cls
    result = np.zeros(len(cat2id) + 1)
    result[cat2id.get(category, len(cat2id))] = 1
    return result
```
Применим преобразование к датасету:
```python
dataset = file_dataset.map(
    lambda img_file_tensor, ann_file_tensor:
        (img_file_tensor, tf.py_func(ann_file2one_hot, [ann_file_tensor], tf.float64))
)
```
Метод `map` возвращает новый датасет, в котором к каждомй строчке изначального
датасета применена функция. Функция на самом деле не приеняется, пока мы
не начали итерироваться по результирующему датасету.
Так же можно заметить, что мы завернули нашу функцию в `tf.py_func` нужно это
т.к. в качестве параметров в функцию преобразования попадают тензоры, а не
те значения, которые в них лежат.
И чтобы работать со стороками нужна эта обретка.

### Загружаем изображение
В Tensorflow есть богатая [библиотека для работы с изображениями][tf_image].
Воспользуемся ею для их загрузки. Нам нужно: прочитать файл, декодировать его
в матрицу, привести матрицу к стандартному размеру (например среднему),
нормализовать значения
в этой матрице.
```python
def image_parser(file_name):
    image_data = tf.read_file(file_name)
    image_parsed = tf.image.decode_jpeg(image_data, channels=3)
    image_parsed = tf.image.resize_image_with_crop_or_pad(image_parsed, 482, 415)
    image_parsed = tf.cast(image_parsed, dtype=tf.float16)
    image_parsed = tf.image.per_image_standardization(image_parsed)
    return image_parsed
```
В отличие от предыдущей функции, здесь `file_name` это тензор, а значит 
нам не надо эту функцию заворачивать, добавим ее в предыдущий сниппет:
```python
dataset = file_dataset.map(
    lambda img_file_tensor, ann_file_tensor:
        (
            image_parser(img_file_tensor),
            tf.py_func(ann_file2one_hot, [ann_file_tensor], tf.float64)
        )
)
```
Проверим что наш граф вычеслений призводит что-то осмысленное:
```python
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()
    print type(next_elem[0])
    with tf.Session() as sess:
        for i in range(3):
            element = sess.run(next_elem)
            print i, element[0].shape, element[1].shape
```
Должно получиться:
```text
0 (482, 415, 3) (201,)
1 (482, 415, 3) (201,)
2 (482, 415, 3) (201,)
```
Как правило в самом начале следовало бы разделить датасет на 2 или 3 части для
треннировки/валидации/тестирования. Мы же воспользуемся разделением на
тренировочный и валидационный датасет из скачанного архива.

Конструирование графа вычислений
================================
Мы будем треннировать сверточную нейронную сеть (англ. convolutional neural netwrok, CNN)
методом похожим на [стохастический градиентный спуск][SGD], но будем использовать
улучшеную его версию [Adam][adam_optimizer]. Для этого нам надо объединить
наши экземпляры в "пакеты" (англ. batch). Кроме того чтобы утилизировать
многопроцессорность (а в лучшем случае наличие GPU для обучения) можно включить
фоновую подкачку данных
```python
BATCH_SIZE = 16
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(2) 
```
Будем объдинять в пакеты по `BATCH_SIZE` экземпляров и подкачивать 2 таких пакета.



Цикл тренировки и валидации
===========================
  * Эпохи
  * Перееинициализация итераторов

Сохранение графа и использование
================================

[1]: https://www.tensorflow.org/
[2]: http://www.image-net.org/
[3]: https://www.kaggle.com/c/imagenet-object-detection-challenge
[python_org]: https://www.python.org/
[install_tensorflow]: https://www.tensorflow.org/install/
[download_dataset]:https://www.kaggle.com/c/imagenet-object-detection-challenge/data
[tf_dataset]: https://www.tensorflow.org/versions/master/api_guides/python/input_dataset
[one_hot_rus_wiki]: https://ru.wikipedia.org/wiki/Унитарный_код
[beautiful_soup]: https://pypi.python.org/pypi/beautifulsoup4
[tf_image]: https://www.tensorflow.org/api_docs/python/tf/image
[SGD]: https://habrahabr.ru/company/ods/blog/326418/#stohasticheskiy-gradientnyy-spusk-i-onlayn-podhod-k-obucheniyu
[adam_optimizer]: https://arxiv.org/abs/1412.6980