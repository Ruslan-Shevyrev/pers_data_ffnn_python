import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import data
from config import config
import utils


def get_study_datasets():
    print('Starting to prepare data')

    input_data, output_data = data.get_study_data()

    input_train_data, input_test_data, output_train_data, output_test_data = \
        (train_test_split(input_data, output_data, test_size=config.TEST_SIZE))

    train_dataset = tf.data.Dataset.from_tensor_slices((input_train_data, output_train_data))
    train_dataset = train_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((input_test_data, output_test_data))
    test_dataset = test_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def learn(model, train_dataset, test_dataset):
    print('Starting to train model')
    model.fit(train_dataset,
              epochs=config.EPOCHS,
              validation_data=test_dataset,
              validation_steps=config.VALIDATION_STEPS)

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)


def training_new_model(output_model):

    train_dataset, test_dataset = get_study_datasets()

    print('Starting to create model')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4096, input_shape=[config.INPUT_SIZE]),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(2048),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(config.OUTPUT_SIZE, activation=tf.keras.activations.softmax)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
                  metrics=['accuracy'])

    learn(model, train_dataset, test_dataset)

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss 1:', test_loss)
    print('Test Accuracy 1:', test_acc)

    model.save(output_model)


def training_exists_model(input_model, output_model):
    train_dataset, test_dataset = get_study_datasets()
    model = tf.keras.models.load_model(input_model)

    learn(model, train_dataset, test_dataset)

    model.save(output_model)


def convert_to_h5(input_model):
    print('Load_model')
    model = tf.keras.models.load_model(input_model)
    model.save(input_model+'.h5')


def predict(model_name, predict_data):

    predict_data = list(map(utils.str_to_arr, predict_data))

    predict_data = tf.stack(predict_data)

    model = tf.keras.models.load_model(model_name)
    return list(map(np.argmax, model.predict(predict_data)))


def test_predict(model_name):
    predict_data = data.get_predict_data()
    predicts = predict(model_name, predict_data)
    for index, value in enumerate(predicts):
        print(predict_data[index])
        print('Predict: ' + str(value))
