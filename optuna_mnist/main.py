import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import optuna


def objective(trial):
    """ Loading Dataset """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    """ Split Dataset in new Train and Test"""
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    train_size = 0.7 # modify train size
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=1337)

    """ Normailize Data to 0 - 1 """
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    """ Model Creation """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    n_layers = trial.suggest_int("n_layers", 1, 4)

    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 32, 256, log=True)
        model.add(tf.keras.layers.Dense(units=num_hidden, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    """ Choose Optimizer and Loss Function """
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])  # modify optimizer
    model.compile(optimizer=optimizer_name, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    """ Training """
    n_epochs = trial.suggest_int("n_epochs", 3, 8)
    model.fit(x_train, y_train, epochs=n_epochs)

    """ Evaluation """
    loss, accuracy = model.evaluate(x_test, y_test)
    print(accuracy)
    print(loss)

    return loss


# possibilities: 4 layers x
if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.show()



