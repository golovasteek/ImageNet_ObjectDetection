О чем эта статья
================
Глубокие нейронные сети сейчас модная тема.
В Сети много тюториалов и видеолекций, и других материалов обсуждающих 
основные принципы, архитектуру, стратегии обучения и т.д.
При этом большинство руководств обходят стороной вопросы загрузки
данных для обучения и сохранения модели, для последующиего использования.
Я хочу рассказать о том, как загружать данные при треннировке нейронных
сетей с использованием [Tensorflow][1]. Если вы еще не пробовали работать с этой библиотекой,
возможно стоит изучить основные концепции, например в статье
[Библиотека глубокого обучения Tensorflow][habr_tf_overview], или на [официальном сайте][1]

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

Традиционные алиасы для библиотек:

```python
import tensorflow as tf
import numpy as np
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

В ходе обучения мы хотим переодически прогонять валидацию, на выборке, которая не учасвует в
обучени. А значит нам надо повторить все манипуляции выше для еще одного датасета.
К счастью всех их можно объединить в функцию например `dataset_from_file_iterator` и создать два 
датасета:
```python
train_dataset = dataset_from_file_iterator(
    functools.partial(image_annotation_iterator, "./ILSVRC", subset="train"),
    cat2id,
    BATCH_SIZE
)
valid_dataset = ... # то же самое только subset="val"
```

Но так как мы хотим дальше использовать один и тот же граф вычислений для тренировки и валидации,
мы создадим более гибкий итератор. Такой который позволяет его переинициализировать.
```python
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )

    train_initializer_op = iterator.make_initializer(train_dataset)
    valid_initializer_op = iterator.make_initializer(valid_dataset)
```
Позже "выполнив" ту или иную операцию мы сможем переключать итератор с одного датасета на 
другой.
```python
with tf.Session(config=config, graph=graph) as sess:
    sess.run(train_initialize_op)
    # Треннируем
    # ...
    
    sess.run(valid_initialize_op)
    # валидируем
    # ...
```
Для тепреь нам нужно описать нашу нейронную сеть, но не будем углубляться в этот вопрос.
Будем считать что функция `semi_alex_net_v1(mages_batch, num_labels)` строит нужную архитекутуру и 
возвращает тензор с выходными значениями, предсказанными нейронной сетью.

Зададим функцию ошибки, и тончности, операцию оптимизации:
```python
img_batch, label_batch = iterator.get_next()

logits = semi_alexnet_v1.semi_alexnet_v1(img_batch, len(cat2id))
loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=label_batch)

labels = tf.argmax(label_batch, axis=1)
predictions = tf.argmax(logits, axis=1)

correct_predictions = tf.reduce_sum(tf.to_float(tf.equal(labels, predictions)))
optimizer = tf.train.AdamOptimizer().minimize(loss)
```

Цикл тренировки и валидации
===========================
Теперь можно приступить к обучению:
```python
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_initializer_op)
    counter = tqdm()
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

```
Выше мы создаем сессию, инициализируем глобальные и локальные переменные в графе, инициализируем 
итератор тренировочными данными. [tqdm][tgdm] не относится к процессу обучения, это просто
удобный инструмент визуализации прогресса.

В контексте той же сессии запускаем и валидацию: цикл валидации выглядит очень похоже. Основная
разница: не запускается операция оптимизации.
```python
with tf.Session() as sess:
    # Train
    # ...
    
    # Validate
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
```
### Эпохи и чекпойнты
Одного простого прохода по всем изображениям конечно же не достаточно для треннировки. И нужно
код треннировки и валидации выше выполнять в цикле (внутри одной сессии).
Выполнять либо фиксированное число итераций, либо пока обучение помогает. Один проход по всему
набору данных традиционно называется эпохой (англ. epoch).

На случай непредвиденных остановок обучения и для дальнейшего использования модели, нужно ее
сохранять. Для этого при создании графа выполнения нужно создать объект класса `Saver`. А в ходе 
тренировки сохранять сотояние модели.
```python
# создаем граф
# ...
saver = tf.train.Saver()

# Создаем сессию
with tf.Session() as sess:
    for i in range(EPOCHS):
        # Train
        # ...
        
        # Validate
        # ...
        saver.save(sess, "checkpoint/name")

```

Что дальше
==========
Мы научились создавать датасеты, преобразовывать их с использованием функций работы с 
тензорами, а так же обычными функциями написанными на питоне. Научились загружать изображения
в фоновом цикле не пытаясь загрузить их в память или сохранить в разжатом виде.
Так же научились сохранять обученную модель.
Применив часть шагов из описанных выше и [загрузив][restore_model] ее можно сделать программу
которая будет распознавать изображения.

В статье совершенно не раскрывается тем нейронных сетей как таковых, их архитектуре и методов
обучение. Для тех кто хочет в этом разобрраться могу порекомендвать курс
[Deep Learning by Google][deep_learing_ud] на Udacity, он подойдет в том числе и совсем
новичкам, без серьёзного бэкграунда. Про применение сверточных нейронных сетей для распознавания
есть отличный курс лекций [Convolutional Neural Networks for Visual Recognition][stanford_lectures]
от Стэнфордского университета. Так же стоит посмотреть на курсы специализации
[Deep Learning][coursera] на Сoursera. Так же есть довольно много материалов на Хабрахабр,
например неплохой обзор библиотеки [Tensorflow][habr_tf_overview] от Open Data Science.

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
[tqdm]: https://pypi.python.org/pypi/tqdm
[restore_model]: https://www.tensorflow.org/api_docs/python/tf/train/Saver#restore
[deep_learing_ud]: https://www.udacity.com/course/deep-learning--ud730
[stanford_lectures]: https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
[coursera]: https://www.coursera.org/specializations/deep-learning
[habr_tf_overview]: https://habrahabr.ru/company/ods/blog/324898/