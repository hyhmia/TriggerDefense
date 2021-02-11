import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.applications import ResNet50
import cv2 as cv
import numpy as np
from tqdm import tqdm
from ModelUtil import *
from dataLoader import *
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)

# DATA_NAME = sys.argv[1]
# MODEL = "ResNet50"
# a=float(sys.argv[2])
# EPOCHS = int(sys.argv[3])
DATA_NAME = "CIFAR"
MODEL = "ResNet50"
a=0.5
EPOCHS = 50

BASE = 10
BATCH_SIZE = 64
LEARNING_RATE_gen = 5e-3
LEARNING_RATE_disc = 5e-5
BASE_WEIGHTS = f'weights/Baseline/{DATA_NAME}_{MODEL}_10.hdf5'
WEIGHTS_PATH_GEN = f"weights/Gan_Defense/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.hdf5"
WEIGHTS_PATH_DISC = f"weights/Gan_Defense/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.hdf5"
(x_train, y_train), (x_test, y_test), _= globals()['load_' + DATA_NAME]("Target")

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

def make_detector_model(input_shape):
    model = tf.keras.Sequential([
        ResNet50(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        layers.GlobalAveragePooling2D()
    ])
    return model
def make_generator_model(input_shape):
    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*512, use_bias=False, input_shape = input_shape))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model
def blend_noise(x_, trigger, alpha):
    noise = tf.squeeze(trigger)
    return tf.map_fn(fn=lambda x : x*(1-alpha)+noise*alpha, elems=x_)

baseline = tf.keras.models.load_model(BASE_WEIGHTS)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_disc)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_ori_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()
img = cv.imread('hellokitty.jpeg')
base_img = tf.convert_to_tensor(np.asarray([img/255]), dtype=tf.float32)
generator = make_generator_model((2048,))
detector = make_detector_model(base_img.shape[1:])
base_noise = detector(base_img)
alpha = tf.constant(a, dtype=tf.float32)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(base_noise, training=True)
        inputs = generated_image * alpha + images*(1-alpha)
        outputs = baseline(inputs, training=True)
        ori_loss = loss_fn(labels, baseline(images, training=False))
        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(generated_image))
        disc_loss = loss_1 - 1e-12 * ori_loss
        gen_loss = loss_1 + 1e-6 * loss_2

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, baseline.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, baseline.trainable_variables))
    train_acc_metric.update_state(labels, outputs)
    overall_loss.update_state(labels, outputs)

def train():
    test_acc = 0
    for epoch in range(EPOCHS):
        with tqdm(enumerate(train_dataset)) as tBatch:
            for step, (images, labels) in tBatch:
                train_step(images, labels)
                tBatch.set_description(f"Alpha: {a} Epoch:{epoch + 1}, Step:{step}, Training Loss:{overall_loss.result()}, "
                                       f"Training Acc:{train_acc_metric.result()}")
            train_acc_metric.reset_states()
            overall_loss.reset_states()

        for images, labels in test_dataset:
            generated_image = generator(base_noise, training=False)
            images_test = blend_noise(images, generated_image, alpha)
            y_pred = baseline(images_test, training=False)
            y_pred_ori = baseline(images*(1-alpha), training=False)
            test_ori_acc_metric.update_state(labels, y_pred_ori)
            test_acc_metric.update_state(labels, y_pred)
        tf.print(f"Acc with noise: {test_acc_metric.result()}, Acc without noise: {test_ori_acc_metric.result()}")
        if test_acc_metric.result() > test_acc:
            test_acc = test_acc_metric.result()
            baseline.save(WEIGHTS_PATH_DISC)
            generator.save(WEIGHTS_PATH_GEN)
            print(f"Acc improved to {test_acc}, Gen saved to {WEIGHTS_PATH_GEN}, Disc saved to {WEIGHTS_PATH_DISC}")

        test_ori_acc_metric.reset_states()
        test_acc_metric.reset_states()

train()
