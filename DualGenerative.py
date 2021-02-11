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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)

DATA_NAME = "CIFAR"
MODEL = "ResNet50"
a=0.5
EPOCHS = 30
BASE = 10

# DATA_NAME sys.argv[1]
# MODEL = "ResNet50"
# a=float(sys.argv[2])
# EPOCHS = int(sys.argv[3])
# BASE = int(sys.argv[4])
# lam_2 = float(sys.argv[5])
BATCH_SIZE = 64
LEARNING_RATE_gen = 1e-3
LEARNING_RATE_disc = 5e-5
RANDOM_SEED = 2021
BASE_WEIGHTS = f'weights/Baseline/{DATA_NAME}_{MODEL}_{BASE}.hdf5'
WEIGHTS_PATH_GEN = f"weights/Gan_Duel/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}_{RANDOM_SEED}.hdf5"
WEIGHTS_PATH_DISC = f"weights/Gan_Duel/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}_{RANDOM_SEED}.hdf5"
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


def make_generator_model(input_shape):
    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*512, use_bias=True, input_shape = input_shape))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(center=False))
    model.add(layers.LeakyReLU())
    if DATA_NAME == "CH_MNIST" or DATA_NAME =="New_CIFAR":
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(center=False))
        model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

def blend_noise(x_, trigger, a):
    noise = tf.squeeze(trigger)
    return tf.map_fn(fn=lambda x : (1-a)*x+a*noise, elems=x_), \
           tf.map_fn(fn=lambda x : (1+a)*x-a*noise, elems=x_)


baseline = creat_duel_model(x_train.shape[1:], y_train.shape[1], BASE_WEIGHTS)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_disc)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_ori_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_alpha_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_ori_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_alpha_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()
img = cv.imread('hellokitty.jpeg')
base_img = tf.convert_to_tensor(np.asarray([img/255]), dtype=tf.float32)
# base_img = tf.expand_dims(tf.random.uniform(shape=x_train.shape[1:], seed=RANDOM_SEED), 0)
generator = make_generator_model((2048,))
detector = make_detector_model(base_img.shape[1:])
base_noise = detector(base_img)
alpha = tf.constant(a, tf.float32)

train_, train_ori, train_alpha = [], [], []
test_, test_ori, test_alpha = [], [] ,[]

@tf.function
def train_step(images, labels, base_noise):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(base_noise)
        inputs_A = (1-alpha)*images + alpha*generated_image
        inputs_B = (1+alpha)*images + (-alpha)*generated_image
        outputs = baseline((inputs_A, inputs_B))
        ori_loss = loss_fn(labels, baseline((images, images), training=False))
        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(generated_image))
        disc_loss = loss_1 - 1e-12 * ori_loss
        # gen_loss = loss_1 + 1e-1 * loss_2
        gen_loss = loss_1 + 1e-6*loss_2

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, baseline.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, baseline.trainable_variables))
    train_acc_metric.update_state(labels, outputs)
    train_ori_acc_metric.update_state(labels, baseline((images, images)))
    train_alpha_acc_metric.update_state(labels, baseline(((1-alpha)*images, (1+alpha)*images)))
    overall_loss.update_state(labels, outputs)


def train():
    test_acc = 0
    for epoch in range(EPOCHS):
        with tqdm(enumerate(train_dataset)) as tBatch:
            for step, (images, labels) in tBatch:
                train_step(images, labels, base_noise)
                tBatch.set_description("Epoch:{}, Step:{}, Training Loss:{:.5f}, Training Acc:{:.5f}, Training Alpha Acc: {:.5f}"
                                       .format(epoch + 1, step, overall_loss.result(), train_acc_metric.result()
                                               , train_alpha_acc_metric.result()))
            train_.append(train_acc_metric.result().numpy())
            train_ori.append(train_ori_acc_metric.result().numpy())
            train_alpha.append(train_alpha_acc_metric.result().numpy())
            train_acc_metric.reset_states()
            train_ori_acc_metric.reset_states()
            train_alpha_acc_metric.reset_states()
            overall_loss.reset_states()
        for images, labels in test_dataset:
            generated_image = generator(base_noise, training=False)
            images_test = blend_noise(images, generated_image, alpha)
            y_pred = baseline(images_test, training=False)
            y_pred_alpha = baseline(((1-alpha)*images, (1+alpha)*images), training=False)
            y_pred_ori = baseline((images,images), training=False)
            test_alpha_acc_metric.update_state(labels, y_pred_alpha)
            test_ori_acc_metric.update_state(labels, y_pred_ori)
            test_acc_metric.update_state(labels, y_pred)
        tf.print(f"Acc: {test_acc_metric.result()}, Acc ori: {test_ori_acc_metric.result()}"
                 f", Acc alpha: {test_alpha_acc_metric.result()}")
        if test_acc_metric.result() > test_acc:
            test_acc = test_acc_metric.result()
            baseline.save(WEIGHTS_PATH_DISC)
            generator.save(WEIGHTS_PATH_GEN)
            print(f"Acc improved to {test_acc}, Gen saved to {WEIGHTS_PATH_GEN}, Disc saved to {WEIGHTS_PATH_DISC}")

        test_.append(test_acc_metric.result().numpy())
        test_ori.append(test_ori_acc_metric.result().numpy())
        test_alpha.append(test_alpha_acc_metric.result().numpy())
        test_ori_acc_metric.reset_states()
        test_alpha_acc_metric.reset_states()
        test_acc_metric.reset_states()


def evaluate(genPath, discPath, images, labels, alpha):
    discriminator = tf.keras.models.load_model(discPath)
    generator = tf.keras.models.load_model(genPath)
    discriminator.compile(loss="categorical_crossentropy",
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    generated_image = generator(base_noise)
    images_W = blend_noise(images, generated_image, alpha)
    loss_W, accuracy_W = discriminator.evaluate(images_W, labels, verbose=1)
    loss_alpha, accuracy_alpha = discriminator.evaluate(((1-alpha)*images, (1+alpha)*images), labels, verbose=1)
    loss_O, accuracy_O = discriminator.evaluate((images, images), labels, verbose=1)
    # np.save(f"gan_per/{DATA_NAME}/{MODEL}_Epoch_{EPOCHS}_{a}.npy", generated_image.numpy())
    print(f"Trigger: {loss_W, accuracy_W}")
    print(f"Alpha: {loss_alpha, accuracy_alpha}")
    print(f"Ori: {loss_O, accuracy_O}")

    return (loss_alpha, accuracy_alpha ), (loss_W, accuracy_W), (loss_O, accuracy_O)

train()


# items = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# for i in items:
#     PATH_GEN = f"weights/Gan_Duel/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_{BASE}_{RANDOM_SEED}.hdf5"
#     PATH_DISC = f"weights/Gan_Duel/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{i}_{BASE}_{RANDOM_SEED}.hdf5"
#     evaluate(PATH_GEN, PATH_DISC, x_test, y_test, i)
# train_loss, train_acc, train_loss_W, train_acc_w = evaluate(WEIGHTS_PATH_GEN, WEIGHTS_PATH_DISC, x_train, y_train)
# test_loss, test_acc, test_loss_W, test_acc_w = evaluate(WEIGHTS_PATH_GEN, WEIGHTS_PATH_DISC, x_test, y_test)
# print(f"train_loss: {train_loss}, train_acc: {train_acc}, train_loss_W: {train_loss_W}, train_acc_w: {train_acc_w}")
# print(f"test_loss: {test_loss}, test_acc: {test_acc}, test_loss_W: {test_loss_W}, test_acc_w: {test_acc_w}")
