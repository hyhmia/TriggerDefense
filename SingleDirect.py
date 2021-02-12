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
LEARNING_RATE_gen = 1e-3
LEARNING_RATE_disc = 5e-5
BASE_WEIGHTS = f'weights/Baseline/{DATA_NAME}_{MODEL}_10.hdf5'
WEIGHTS_PATH_BASE = f"weights/Gradient_Defense/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.npy"
WEIGHTS_PATH_DISC = f"weights/Gradient_Defense/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.hdf5"
NOISE_PATH = f"Base/Gradient_Defense/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.png"
(x_train, y_train), (x_test, y_test), _= globals()['load_' + DATA_NAME]("Target")

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)


baseline = tf.keras.models.load_model(BASE_WEIGHTS)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_disc)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_ori_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()
img = cv.imread('hellokitty.jpeg')
base_img = tf.image.resize(np.asarray((img-127.5)/127.5), [128, 128])
alpha = tf.constant(a, dtype=tf.float32)
base = tf.Variable(base_img[tf.newaxis, :], trainable=True)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_tape.watch(base)
        inputs = base * alpha + images*(1-alpha)
        outputs = baseline(inputs)
        ori_loss = loss_fn(labels, baseline((images, images), training=False))

        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(base))
        disc_loss = loss_1 - 1e-12 * ori_loss
        gen_loss = loss_1 + 1e-6 * loss_2

    gen_gradients = gen_tape.gradient(gen_loss, base)
    generator_optimizer.apply_gradients([(gen_gradients, base)])
    disc_gradients = disc_tape.gradient(disc_loss, baseline.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, baseline.trainable_variables))
    train_acc_metric.update_state(labels, outputs)
    overall_loss.update_state(labels, outputs)


def train():
    test_acc = 0
    for epoch in range(EPOCHS):
        with tqdm(enumerate(train_dataset)) as tBatch:
            for step, (images, labels) in tBatch:
                train_step(images, labels)
                tBatch.set_description("Epoch:{}, Step:{}, Training Loss:{}, Training Acc:{}"
                                       .format(epoch + 1, step, overall_loss.result(), train_acc_metric.result()))
            train_acc_metric.reset_states()
            overall_loss.reset_states()

        for images, labels in test_dataset:
            generated_image = base
            images_test = images*(1-alpha) + generated_image*alpha
            y_pred = baseline(images_test, training=False)
            # y_pred_ori = baseline((images*(1-alpha), images*(1+alpha)), training=False)
            y_pred_ori = baseline(images*(1-alpha), training=False)
            test_ori_acc_metric.update_state(labels, y_pred_ori)
            test_acc_metric.update_state(labels, y_pred)
        tf.print(f"Acc with noise: {test_acc_metric.result()}, Acc without noise: {test_ori_acc_metric.result()}")
        if test_acc_metric.result() > test_acc:
            test_acc = test_acc_metric.result()
            baseline.save(WEIGHTS_PATH_DISC)
            np.save(WEIGHTS_PATH_BASE, base.numpy())
            tf.keras.preprocessing.image.save_img(NOISE_PATH, tf.squeeze(base * 255), scale=False)
            print(f"Acc improved to {test_acc}, Gen saved to {WEIGHTS_PATH_BASE}, Disc saved to {WEIGHTS_PATH_DISC}")

        test_ori_acc_metric.reset_states()
        test_acc_metric.reset_states()

def evaluate(noisePath, discPath, images, labels, alpha):
    base_noise = np.load(noisePath)
    discriminator = tf.keras.models.load_model(discPath)
    discriminator.compile(loss="categorical_crossentropy",
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    images_W = base_noise * alpha + images*(1-alpha)
    loss_W, accuracy_W = discriminator.evaluate(images_W, labels, verbose=1)
    loss_O, accuracy_O = discriminator.evaluate(images, labels, verbose=1)
    print(f"Trigger:{loss_W, accuracy_W}")
    print(f"Ori:{loss_O, accuracy_O}")

train()
# items = [0.1, 0.3, 0.5, 0.7]
# #
# for i in items:
#     print(f"i: {i}")
#     PATH_GEN = f"weights/Gradient_Duel/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_10.npy"
#     PATH_DISC = f"weights/Gradient_Duel/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_10.hdf5"
#     evaluate(PATH_GEN, PATH_DISC, x_test, y_test, i)
# train_loss, train_acc, train_loss_W, train_acc_w = evaluate(WEIGHTS_PATH_BASE, WEIGHTS_PATH_DISC, x_train, y_train)
# test_loss, test_acc, test_loss_W, test_acc_w = evaluate(WEIGHTS_PATH_BASE, WEIGHTS_PATH_DISC, x_test, y_test)

