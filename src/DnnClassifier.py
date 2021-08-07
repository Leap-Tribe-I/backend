import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.output import get_csv_output
def tensorflow_dnn(data, X_train ,X_test, y_train, y_test, accuracyDict, timelog):
    start = time.time()
    labels = data.pop('suicidal_thoughts')
    ds = tf.data.Dataset.from_tensor_slices((dict(data), labels))

    #converting pandas columns to tensorflow features
    family_size = tf.feature_column.numeric_column("family_size")
    annual_income = tf.feature_column.numeric_column("annual_income")
    eating_habits= tf.feature_column.numeric_column("eating_habits")
    addiction_friend = tf.feature_column.numeric_column("addiction_friend")
    addiction = tf.feature_column.numeric_column("addiction")
    medical_history = tf.feature_column.numeric_column("medical_history")
    depressed = tf.feature_column.numeric_column("depressed")
    anxiety = tf.feature_column.numeric_column("anxiety")
    happy_currently = tf.feature_column.numeric_column("happy_currently")
    feature_columns = [family_size, annual_income, eating_habits, addiction_friend, addiction, medical_history, depressed, anxiety, happy_currently]
    def train_input(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        return dataset.shuffle(1000).repeat().batch(batch_size)

    def eval_input(features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features=dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset


    model = tf.estimator.DNNClassifier(feature_columns=feature_columns,  hidden_units=[50, 40, 10], optimizer= tf.keras.optimizers.Adagrad(), batch_norm=True)
    model.train(input_fn=lambda:train_input(X_train, y_train, 100), steps = 1000)
    result = model.evaluate(input_fn=lambda:eval_input(X_test, y_test, 100))
    accuracy = result['accuracy'] *100
    accuracyDict['DNN Classifier'] = accuracy

    #making arguments to send for OutputCsv
    y_pred_class = []
    #model.predict() function takes the same input function as evaluate
    #the key class_ids stores the corresponding value of the prediciton the model made
    for single_prediction in model.predict(input_fn=lambda:eval_input(X_test, y_test, 50)):
        for key, value in single_prediction.items():
            if key == 'class_ids':
                y_pred_class.append(value[0])
    get_csv_output("DNNClassifier", X_test, y_pred_class)
    end = time.time()
    timelog['DNN Classifier'] = end - start