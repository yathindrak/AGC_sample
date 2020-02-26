import tensorflow as tf

import cyclegan_datasets
import model


def _load_samples(csv_name, image_type):

    tf.compat.v1.disable_eager_execution()

    filename_queue = tf.compat.v1.train.string_input_producer([csv_name])
    # filename_queue = tf.data.Dataset.from_tensor_slices([csv_name])

    reader = tf.compat.v1.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.io.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.io.read_file(filename_i)
    file_contents_j = tf.io.read_file(filename_j)
    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    return image_decoded_A, image_decoded_B


def load_data(dataset_name, image_size_before_crop,
              do_shuffle=True, do_flipping=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    image_i, image_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {
        'image_i': image_i,
        'image_j': image_j
    }

    # Preprocessing:
    inputs['image_i'] = tf.image.resize(
        inputs['image_i'], [image_size_before_crop, image_size_before_crop])
    inputs['image_j'] = tf.image.resize(
        inputs['image_j'], [image_size_before_crop, image_size_before_crop])

    if do_flipping is True:
        inputs['image_i'] = tf.image.random_flip_left_right(inputs['image_i'], seed=1)
        inputs['image_j'] = tf.image.random_flip_left_right(inputs['image_j'], seed=1)

    inputs['image_i'] = tf.image.random_crop(
        inputs['image_i'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3], seed=1)
    inputs['image_j'] = tf.image.random_crop(
        inputs['image_j'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3], seed=1)

    inputs['image_i'] = tf.subtract(tf.math.divide(inputs['image_i'], 127.5), 1)
    inputs['image_j'] = tf.subtract(tf.math.divide(inputs['image_j'], 127.5), 1)

    # tf.compat.v1.disable_v2_behavior()

    # Batch
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
            [inputs['image_i'], inputs['image_j']], 1, 5000, 100, seed=1)
    else:
        inputs['images_i'], inputs['images_j'] = tf.compat.v1.train.batch(
            [inputs['image_i'], inputs['image_j']], 1)

    return inputs