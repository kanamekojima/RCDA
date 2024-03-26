import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Cropping1D,
    Dropout,
    MaxPooling1D,
    UpSampling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2


class ResnetBlock(keras.layers.Layer):

    def __init__(self, filter_size, kernel_size, kernel_regularizer=None):
        super(ResnetBlock, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer

        self.bn_a = BatchNormalization()
        self.conv_a = Conv1D(
            filter_size, kernel_size=1, padding='same')

        self.bn_b = BatchNormalization()
        self.conv_b = Conv1D(
            filter_size, kernel_size=kernel_size, padding='same',
            kernel_regularizer=kernel_regularizer)

        self.bn_c = BatchNormalization()
        self.conv_c = Conv1D(
            filter_size, kernel_size=1, padding='same')

        self.conv_sc = Conv1D(
            filter_size, kernel_size=1, padding='same')

    def call(self, input_tensor, training=False):
        x = self.conv_a(input_tensor)
        x = self.bn_a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_b(x)
        x = self.bn_b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_c(x)
        x = self.bn_c(x, training=training)

        x += self.conv_sc(input_tensor)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            'filter_size': self.filter_size,
            'kernel_size': self.kernel_size,
            'kernel_regularizer': self.kernel_regularizer,
        }
        return dict(list(base_config.items()) + list(config.items()))


class ResnetBlockV2(keras.layers.Layer):

    def __init__(self, filter_size, kernel_size, kernel_regularizer):
        super(ResnetBlockV2, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer

        self.bn_a = BatchNormalization()
        self.conv_a = Conv1D(
            filter_size, kernel_size, padding='same',
            kernel_regularizer=kernel_regularizer)

        self.bn_b = BatchNormalization()
        self.conv_b = Conv1D(
            filter_size, kernel_size, padding='same',
            kernel_regularizer=kernel_regularizer)

        self.conv_sc = Conv1D(
            filter_size, kernel_size=1, padding='same',
            kernel_regularizer=kernel_regularizer)

    def call(self, input_tensor, training=False):
        x = self.bn_a(input_tensor, training=training)
        x1 = tf.nn.relu(x)
        x = self.conv_a(x1)

        x = self.bn_b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_b(x)

        x += self.conv_sc(x1)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            'filter_size': self.filter_size,
            'kernel_size': self.kernel_size,
            'kernel_regularizer': self.kernel_regularizer,
        }
        return dict(list(base_config.items()) + list(config.items()))


class ResnetBottleneckBlockV2(keras.layers.Layer):

    def __init__(self, filter_size, kernel_size, kernel_regularizer=l2(0)):
        super(ResnetBottleneckBlockV2, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer

        filter_size_bottle = filter_size // 4

        self.bn_a = BatchNormalization()
        self.conv_a = Conv1D(
            filter_size_bottle, kernel_size=1, kernel_initializer='he_normal',
            padding='same', kernel_regularizer=kernel_regularizer)

        self.bn_b = BatchNormalization()
        self.conv_b = Conv1D(
            filter_size_bottle, kernel_size, kernel_initializer='he_normal',
            padding='same', kernel_regularizer=kernel_regularizer)

        self.bn_c = BatchNormalization()
        self.conv_c = Conv1D(
            filter_size, kernel_size=1, kernel_initializer='he_normal',
            padding='same', kernel_regularizer=kernel_regularizer)

        self.conv_sc = Conv1D(
            filter_size, kernel_size=1, padding='same',
            kernel_regularizer=kernel_regularizer)

    def call(self, input_tensor, training=False):
        x = self.bn_a(input_tensor, training=training)
        x1 = tf.nn.relu(x)
        x = self.conv_a(x1)

        x = self.bn_b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_b(x)

        x = self.bn_c(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_c(x)

        x += self.conv_sc(x1)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            'filter_size': self.filter_size,
            'kernel_size': self.kernel_size,
            'kernel_regularizer': self.kernel_regularizer,
        }
        return dict(list(base_config.items()) + list(config.items()))


def get_ResNet_model(
        inChannel,
        left_margin=0,
        right_margin=0,
        dropout_rate=0.25,
        kr=1e-4,
        ):
    model = Sequential()

    model.add(ResnetBlock(32, 5, kernel_regularizer=None))
    model.add(MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBlock(64, 5, kernel_regularizer=None))
    model.add(MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBlock(128, 5, kernel_regularizer=None))

    model.add(ResnetBlock(64, 5, kernel_regularizer=None))
    model.add(UpSampling1D(2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBlock(32, 5, kernel_regularizer=None))
    model.add(UpSampling1D(2))
    if left_margin > 0 or right_margin > 0:
        model.add(Cropping1D(cropping=(left_margin, right_margin)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Conv1D(inChannel - 1, 5, activation='softmax', padding='same'))

    return model


def get_ResNetV2_model(
        inChannel,
        left_margin=0,
        right_margin=0,
        dropout_rate=0.25,
        kr=1e-4,
        lr=1e-3,
        ):
    model = Sequential()

    model.add(ResnetBlockV2(32, 5, kernel_regularizer=l1(kr)))
    model.add(MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBlockV2(64, 5, kernel_regularizer=l1(kr)))
    model.add(MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBottleneckBlockV2(128, 5, kernel_regularizer=l1(kr)))

    model.add(ResnetBlockV2(64, 5, l1(kr)))
    model.add(UpSampling1D(2))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(ResnetBlockV2(32, 5, kernel_regularizer=l1(kr)))
    model.add(UpSampling1D(2))
    if left_margin > 0 or right_margin > 0:
        model.add(Cropping1D(cropping=(left_margin, right_margin)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Conv1D(inChannel - 1, 5, activation='softmax', padding='same'))

    return model


def get_SCDA_model(
        num_positions,
        inChannel,
        left_margin=0,
        right_margin=0,
        dropout_rate=0.25,
        kr=1e-4,
        lr=1e-3,
        ):
    model = Sequential()

    model.add(Conv1D(
        32, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(
        64, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(
        128, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))

    model.add(Conv1D(
        64, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))
    model.add(UpSampling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(
        32, 5, padding='same', activation='relu', kernel_regularizer=l1(kr)))
    model.add(UpSampling1D(2))
    if left_margin > 0 or right_margin > 0:
        model.add(Cropping1D(cropping=(left_margin, right_margin)))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(inChannel - 1, 5, activation='softmax', padding='same'))

    return model
