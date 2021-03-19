import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import to_categorical

import os
import quaternion

from keras.optimizers import Adam
import datetime
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def add_first_empty_channel(image, label):
    return tf.concat([tf.zeros(image.shape[:-1] + (1,)), image], axis=-1), label


def label_to_categorical(image, label):
    num_classes = 10
    return image, tf.one_hot(label, num_classes)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, conv_filters=0, dropout_rate=0., max_pool_padding='valid', quaternion_mode_=False):
        super().__init__()
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.max_pool_padding = max_pool_padding
        self.quaternion_mode = quaternion_mode_

        if self.quaternion_mode:
            self.conv_filters /= 2
            self.ConvGeneric = quaternion.QuaternionConv2D
        else:
            self.ConvGeneric = Conv2D
        self.BatchNormGeneric = BatchNormalization

    def build(self, input_shape):

        self.conv1 = self.ConvGeneric(self.conv_filters, kernel_size=(3, 3), input_shape=input_shape, activation='relu',
                                      data_format='channels_last')
        self.bn1 = self.BatchNormGeneric()
        self.conv2 = self.ConvGeneric(self.conv_filters, kernel_size=(3, 3), activation='relu',
                                      data_format='channels_last')
        self.bn2 = self.BatchNormGeneric()
        self.maxPool = MaxPooling2D(pool_size=(2, 2), padding=self.max_pool_padding)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        return self.dropout(self.maxPool(self.bn2(self.conv2(self.bn1(self.conv1(inputs))))))


class ConvModel(tf.keras.Model):
    def __init__(self, quaternion_mode_=False):
        super().__init__()

        self.b1 = ConvBlock(32, 0.2, quaternion_mode_=quaternion_mode_)
        self.b2 = ConvBlock(64, 0.3, quaternion_mode_=quaternion_mode_)
        self.b3 = ConvBlock(128, 0.4, max_pool_padding='same', quaternion_mode_=quaternion_mode_)

        # flattening followed by dense layer and final output layer
        if quaternion_mode:
            self.dense1 = quaternion.QuaternionDense(128, activation='relu')
        else:
            self.dense1 = Dense(128, activation='relu')

        self.dense2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        h = Flatten()(h)
        h = self.dense1(h)
        h = Dropout(0.5)(h)
        outputs = self.dense2(h)
        tf.summary.histogram('outputs', outputs)
        return outputs


tf.keras.backend.set_image_data_format('channels_last')

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Build training pipeline
ds_train = ds_train.map(
    label_to_categorical, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.map(
    add_first_empty_channel, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for i, element in enumerate(ds_train.as_numpy_iterator()):
    if i < 1:
        print(f"x {element[0]}")
        print(f"y {element[1]}")
    else:
        break

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Build evaluation pipeline
ds_test = ds_test.map(
    label_to_categorical, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    add_first_empty_channel, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

x_shape = ds_info.features['image'].shape
y_shape = ds_info.features['label'].shape

parent_dir = "../../cifar10"
log_dir = os.path.join(parent_dir, 'logs')
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

quaternion_mode = True
history = []

for idx, quaternion_mode in enumerate([False, True]):
# for idx, quaternion_mode in enumerate([True]):
    model_type = 'quad' if quaternion_mode else 'trad'
    exp_name = "-".join([model_type, now, 'batchnorm'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, exp_name))

    model = ConvModel(quaternion_mode_=quaternion_mode)
    model(next(ds_train.as_numpy_iterator())[0])
    model.summary()

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Fit the model on train data
    history.append(model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_test,
        callbacks=[tensorboard_callback]
    )
    )

    # save the model in H5 file
    model.save(os.path.join(parent_dir, 'models', exp_name))

import matplotlib.pyplot as plt

for hist in history:
    #  Ploting the accuracy graph
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.show()
    # Ploting the loss graph
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.show()
