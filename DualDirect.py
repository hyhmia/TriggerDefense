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
EPOCHS = 30
BASE = 10
BATCH_SIZE = 64
LEARNING_RATE_gen = 1e-3
LEARNING_RATE_disc = 5e-5
BASE_WEIGHTS = f'weights/Baseline/{DATA_NAME}_{MODEL}_{BASE}.hdf5'
WEIGHTS_PATH_BASE = f"weights/Gradient_Duel/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.npy"
WEIGHTS_PATH_DISC = f"weights/Gradient_Duel/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.hdf5"
NOISE_PATH = f"Base/Gradient_Duel/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.png"
(x_train, y_train), (x_test, y_test), _= globals()['load_' + DATA_NAME]("Target")

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

def creat_parallel_model(input_shape, output_shape, base_weights):
    base_weights = tf.keras.models.load_model(base_weights).layers[0].get_weights()
    left_input = tf.keras.layers.Input(shape=input_shape, name='left_input')
    right_input = tf.keras.layers.Input(shape=input_shape, name='right_input')
    left_model = tf.keras.Sequential([ResNet50(include_top=False,
                 weights=None,
                 input_shape=input_shape), GlobalAveragePooling2D()],
                                     name="left_model")
    right_model = tf.keras.Sequential([ResNet50(include_top=False,
                 weights=None,
                 input_shape=input_shape), GlobalAveragePooling2D()],
                                      name="right_model")
    # left_model = hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x1/1", trainable=True)
    # right_model = hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x1/1", trainable=True)
    concat = tf.keras.layers.Concatenate()([left_model(left_input), right_model(right_input)])
    left_model.set_weights(base_weights)
    right_model.set_weights(base_weights)
    logits = tf.keras.layers.Dense(output_shape)(concat)
    output = tf.keras.layers.Activation("softmax")(logits)
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=[output])
    model.summary()
    return model
def creat_duel_model(input_shape, output_shape, base_weights):
    base_model = tf.keras.models.load_model(base_weights)
    left_input = tf.keras.layers.Input(shape=input_shape, name='left_input')
    right_input = tf.keras.layers.Input(shape=input_shape, name='right_input')
    input_model = tf.keras.Sequential([ResNet50(include_top=False,
                 weights=None,
                 input_shape=input_shape), GlobalAveragePooling2D()], name="model")
    concat = tf.keras.layers.Concatenate()([input_model(left_input), input_model(right_input)])
    input_model.set_weights(base_model.layers[0].get_weights())
    logits = tf.keras.layers.Dense(output_shape)(concat)
    output = tf.keras.layers.Activation("softmax")(logits)
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=[output])
    model.layers[-2].set_weights([np.tile(base_model.layers[-2].get_weights()[0], (2, 1)), base_model.layers[-2].get_weights()[1]])
    model.summary()
    return model
def blend_noise(x_, trigger, alpha):
    noise = tf.squeeze(trigger)
    return tf.map_fn(fn=lambda x : x*(1-alpha)+noise*alpha, elems=x_), \
           tf.map_fn(fn=lambda x : x*(1+alpha)+noise*(-alpha), elems=x_)

baseline = creat_duel_model(x_train.shape[1:], y_train.shape[1], BASE_WEIGHTS)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_disc)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_ori_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()
img = cv.imread('hellokitty.jpeg')
base_img = tf.image.resize(np.asarray(img/255), [128, 128])
# base_img = tf.zeros([32, 32, 3], tf.float32)
alpha = tf.constant(a, dtype=tf.float32)
base = tf.Variable(base_img[tf.newaxis, :], trainable=True)
# print(base.shape)
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_tape.watch(base)
        inputs_A = base * alpha + images*(1-alpha)
        inputs_B = base * (-alpha) + images * (1 + alpha)
        # inputs_A = base + images
        # inputs_B = -base + images
        outputs = baseline((inputs_A, inputs_B))
        ori_loss = loss_fn(labels, baseline((images, images), training=False))

        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(base))
        # loss_3 = tri_loss - ori_loss
        disc_loss = loss_1 - 1e-12 * ori_loss
        gen_loss = loss_1 + 1e-8 * loss_2

        # tf.print(f"gen_loss: {gen_loss} disc_loss: {disc_loss}, loss_1: {loss_1}, Loss_2: {loss_2}")
    gen_gradients = gen_tape.gradient(gen_loss, base)
    # tf.print(gen_gradients.shape)
    # tf.print(base.shape)
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
            images_test = blend_noise(images, generated_image, alpha)
            y_pred = baseline(images_test, training=False)
            # y_pred_ori = baseline((images*(1-alpha), images*(1+alpha)), training=False)
            y_pred_ori = baseline((images, images), training=False)
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
    images_W = blend_noise(images, base_noise, alpha)
    loss_W, accuracy_W = discriminator.evaluate(images_W, labels, verbose=1)
    # loss_O, accuracy_O = discriminator.evaluate((images, images), labels, verbose=1)
    loss, accuracy = discriminator.evaluate((images*(1-alpha), images*(1+alpha)), labels, verbose=1)
    print(f"Trigger:{loss_W, accuracy_W}")
    # print(f"Ori:{loss_O, accuracy_O}")
    print(f"alpha: {loss, accuracy}")

# items = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# for i in items:
#     print(f"Gradient i: {i}")
#     PATH_GEN = f"weights/Gradient_Duel/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_10.npy"
#     PATH_DISC = f"weights/Gradient_Duel/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_10.hdf5"
#     evaluate(PATH_GEN, PATH_DISC, x_test, y_test, i)