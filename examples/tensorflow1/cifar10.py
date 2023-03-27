import argparse
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar10
import random

import src.deprecated.error_simulator_tf1 as simulator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convolutional neural network on CIFAR-10 dataset.")
    parser.add_argument("-m", "--mode", nargs=1,
                        default="testing", type=str,
                        choices=["training", "testing", "benchmarking", "graph"],
                        help="Mode to execute.")
    arguments = parser.parse_args()
    selected_mode = vars(arguments)["mode"]
    if type(selected_mode) is list:
        selected_mode = selected_mode[0]
    return selected_mode


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    one_hot_encoder = OneHotEncoder(categories="auto", dtype=np.float32, sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape((-1, 1)))
    y_test = one_hot_encoder.fit_transform(y_test.reshape((-1, 1)))
    cifar10_means = np.mean(x_train, axis=(0, 2, 3))
    cifar10_stds = np.std(x_train, axis=(0, 2, 3))
    for channel in range(0, 3):
        x_train[:, channel, :, :] = (x_train[:, channel, :, :] - cifar10_means[channel]) / cifar10_stds[channel]
        x_test[:, channel, :, :] = (x_test[:, channel, :, :] - cifar10_means[channel]) / cifar10_stds[channel]
    validation_percentage = 0.3
    validation_size = np.floor(float(x_train.shape[0]) * validation_percentage).astype("int")
    validation_indexes = np.random.choice(x_train.shape[0], validation_size, replace=False)
    x_val = x_train[validation_indexes, :, :, :]
    y_val = y_train[validation_indexes, :]
    x_train = np.delete(x_train, validation_indexes, axis=0)
    y_train = np.delete(y_train, validation_indexes, axis=0)
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    X = tf.placeholder(tf.float32, [None, 3, 32, 32])
    Y = tf.placeholder(tf.float32, [None, 10])
    l2_regularizer = tf.contrib.layers.l2_regularizer(5 * 1E-05)
    with tf.variable_scope("conv_1"):
        conv_1 = tf.layers.Conv2D(32, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(X)
    with tf.variable_scope("batchnorm_1"):
        batch_norm_1 = tf.layers.BatchNormalization(axis=1)(conv_1)
    with tf.variable_scope("conv_2"):
        conv_2 = tf.layers.Conv2D(32, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(batch_norm_1)
    with tf.variable_scope("batchnorm_2"):
        batch_norm_2 = tf.layers.BatchNormalization(axis=1)(conv_2)
    with tf.variable_scope("maxpool_1"):
        max_pool_1 = tf.layers.MaxPooling2D(2, 2, padding="same", data_format="channels_first")(batch_norm_2)
    with tf.variable_scope("dropout_1"):
        dropout_1 = tf.layers.Dropout(rate=0.3)(max_pool_1)
    with tf.variable_scope("conv_3"):
        conv_3 = tf.layers.Conv2D(64, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(dropout_1)
    with tf.variable_scope("batchnorm_3"):
        batch_norm_3 = tf.layers.BatchNormalization(axis=1)(conv_3)
    with tf.variable_scope("conv_4"):
        conv_4 = tf.layers.Conv2D(64, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(batch_norm_3)
    with tf.variable_scope("batchnorm_4"):
        batch_norm_4 = tf.layers.BatchNormalization(axis=1)(conv_4)
    with tf.variable_scope("maxpool_2"):
        max_pool_2 = tf.layers.MaxPooling2D(2, 2, padding="same", data_format="channels_first")(batch_norm_4)
    with tf.variable_scope("dropout_2"):
        dropout_2 = tf.layers.Dropout(rate=0.)(max_pool_2)
    with tf.variable_scope("conv_5"):
        conv_5 = tf.layers.Conv2D(128, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(dropout_2)
    with tf.variable_scope("batchnorm_5"):
        batch_norm_5 = tf.layers.BatchNormalization(axis=1)(conv_5)
    with tf.variable_scope("conv_6"):
        conv_6 = tf.layers.Conv2D(128, [3, 3],
                                  padding="same",
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.glorot_uniform_initializer,
                                  kernel_regularizer=l2_regularizer,
                                  data_format="channels_first")(batch_norm_5)
    with tf.variable_scope("batchnorm_6"):
        batch_norm_6 = tf.layers.BatchNormalization(axis=1)(conv_6)
    with tf.variable_scope("maxpool_3"):
        max_pool_3 = tf.layers.MaxPooling2D(2, 2, padding="same", data_format="channels_first")(batch_norm_6)
    with tf.variable_scope("dropout_3"):
        dropout_3 = tf.layers.Dropout(rate=0.5)(max_pool_3)
    flatten = tf.reshape(dropout_3, [-1, 4 * 4 * 128])
    with tf.variable_scope("dense_1"):
        dense_1 = tf.layers.Dense(576, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer)(flatten)
    with tf.variable_scope("dropout_4"):
        dropout_4 = tf.layers.Dropout(rate=0.5)(dense_1)
    with tf.variable_scope("dense_2"):
        logits = tf.layers.Dense(10, kernel_initializer=tf.glorot_uniform_initializer)(dropout_4)
    with tf.variable_scope("output"):
        output = tf.nn.softmax(logits)
    return X, Y, logits, output, batch_norm_1, max_pool_3, flatten, dense_1, conv_1


def train(dataset, model):
    loss_operator = tf.reduce_mean(tf.losses.softmax_cross_entropy(model[1], model[2]))
    l2_loss = tf.losses.get_regularization_loss()
    loss_operator += l2_loss
    learning_rate_placeholder = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
    minimize_operation = optimizer.minimize(loss_operator)
    correct_predictions = tf.equal(tf.argmax(model[1], 1), tf.argmax(model[3], 1))
    accuracy_operator = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    accuracy_batch = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
    batch_size = 64
    steps = 5000
    learning_rate = 0.001
    best_accuracy = 0
    best_loss = 1E100
    memory_configuration = tf.ConfigProto()
    memory_configuration.gpu_options.allow_growth = True
    with tf.Session(config=memory_configuration) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for step in range(steps + 1):
            x_train_batch, y_train_batch = [], []
            for i in range(batch_size):
                index = random.randint(0, dataset[0].shape[0] - 1)
                x_train_batch.append(dataset[0][index, :, :, :])
                y_train_batch.append(dataset[1][index, :])
            if step == 2500:
                learning_rate /= 10.0
            if step == 3000:
                learning_rate /= 10.0
            session.run(minimize_operation,
                        feed_dict={
                            model[0]: x_train_batch,
                            model[1]: y_train_batch,
                            learning_rate_placeholder: learning_rate})
            if step % 100 == 0:
                loss, accuracy = session.run([loss_operator, accuracy_operator],
                                             feed_dict={model[0]: x_train_batch, model[1]: y_train_batch})
                validation_batches = np.ceil(float(dataset[4].shape[0]) / float(batch_size)).astype("int")
                total_correct = 0
                total_loss = 0.0
                for validation_batch in range(validation_batches):
                    if validation_batch == validation_batches - 1:
                        x_validation_batch = dataset[4][validation_batch * batch_size:, :, :, :]
                        y_validation_batch = dataset[5][validation_batch * batch_size:, :]
                    else:
                        x_validation_batch = dataset[4][
                                             validation_batch * batch_size: (validation_batch + 1) * batch_size, :, :,
                                             :]
                        y_validation_batch = dataset[5][
                                             validation_batch * batch_size: (validation_batch + 1) * batch_size, :]
                    batch_loss, batch_correct = session.run([loss_operator, accuracy_batch],
                                                            feed_dict={model[0]: x_validation_batch,
                                                                       model[1]: y_validation_batch})
                    total_correct += batch_correct
                    total_loss += batch_loss
                validation_accuracy = float(total_correct) / float(dataset[4].shape[0])
                validation_loss = total_loss / float(validation_batches)
                if validation_accuracy >= best_accuracy and validation_loss <= best_loss:
                    best_accuracy = validation_accuracy
                    best_loss = validation_loss
                    saver.save(session, "cifar10modelnamed")
                    print("Model saved.")
                print(
                    "Step: {:0>4d}\tTr. loss: {:.5f}\tTr. acc: {:.4f}\t\tVal loss: {:.5f}\tVal acc: {:.4f}".format(step,
                                                                                                                   loss,
                                                                                                                   accuracy,
                                                                                                                   validation_loss,
                                                                                                                   validation_accuracy))


def test(dataset, model):

    instances = [
        ('Conv2D3x3', ('conv_6/conv2d/Conv2D', '(?, 128, 8, 8)')),
        ('Conv2D3x3', ('conv_1/conv2d/Conv2D', '(?, 32, 32, 32)')),
        ('Conv2D3x3', ('conv_3/conv2d/Conv2D', '(?, 64, 16, 16)')),
        ('Conv2D3x3', ('conv_5/conv2d/Conv2D', '(?, 128, 8, 8)')),
        ('Conv2D3x3', ('conv_2/conv2d/Conv2D', '(?, 32, 32, 32)')),
        ('Conv2D3x3', ('conv_4/conv2d/Conv2D', '(?, 64, 16, 16)')),
        ('BiasAdd', ('conv_1/conv2d/BiasAdd', '(?, 32, 32, 32)')),
        ('BiasAdd', ('dense_1/dense/BiasAdd', '(?, 576)')),
        ('BiasAdd', ('conv_6/conv2d/BiasAdd', '(?, 128, 8, 8)')),
        ('BiasAdd', ('dense_2/dense/BiasAdd', '(?, 10)')),
        ('BiasAdd', ('conv_4/conv2d/BiasAdd', '(?, 64, 16, 16)')),
        ('BiasAdd', ('conv_3/conv2d/BiasAdd', '(?, 64, 16, 16)')),
        ('BiasAdd', ('conv_2/conv2d/BiasAdd', '(?, 32, 32, 32)')),
        ('BiasAdd', ('conv_5/conv2d/BiasAdd', '(?, 128, 8, 8)')),
        ('FusedBatchNorm', ('batchnorm_2/batch_normalization/FusedBatchNorm', '(?, 32, 32, 32)')),
        ('FusedBatchNorm', ('batchnorm_6/batch_normalization/FusedBatchNorm', '(?, 128, 8, 8)')),
        ('FusedBatchNorm', ('batchnorm_5/batch_normalization/FusedBatchNorm', '(?, 128, 8, 8)')),
        ('FusedBatchNorm', ('batchnorm_4/batch_normalization/FusedBatchNorm', '(?, 64, 16, 16)')),
        ('FusedBatchNorm', ('batchnorm_1/batch_normalization/FusedBatchNorm', '(?, 32, 32, 32)')),
        ('FusedBatchNorm', ('batchnorm_3/batch_normalization/FusedBatchNorm', '(?, 64, 16, 16)')),
    ]


    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, "cifar10modelnamed")

        error_sim = simulator.FaultInjector(session)

        indexes = {
            0: [10]
            }
        
        repetitions = 10000

        for instance in instances:
            for golden_digit in indexes.keys():  
                for index in indexes[golden_digit]:
                    feed_dict = {
                        model[0]: dataset[2][index, :, :, :].reshape((1, 3, 32, 32))
                    }

                    error_sim.instrument([model[3]], feed_dict)
                    errors = 0
                    for _ in range(repetitions):
                        # while True:
                        res = error_sim.generate_injection_sites('OPERATOR_SPECIFIC', 1, instance[0],
                                                                 op_instance=instance[1])

                        if not res:
                            continue
                        
                        results, _, _ = error_sim.inject(
                            [model[3]],
                            feed_dict)
                        
                        correct_result = golden_digit == np.argmax(results[0][0])
                        if not correct_result:
                            errors += 1
            print(f'Instance {instance[1][0]} got {str(errors)} misclassifications over {str(repetitions)} simulations')
                        

if __name__ == "__main__":
    # mode = parse_arguments()

    mode = 'testing'
    print("Mode: {}.".format(mode))

    dataset = load_cifar10_dataset()
    print("Cifar 10 dataset has been loaded and pre-processed.")

    model = build_model()
    print("The model has been built.")

    if mode == "training":
        train(dataset, model)
    elif mode == "testing":
        test(dataset, model)
