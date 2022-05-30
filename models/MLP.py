import tensorflow as tf
import pandas as pd
import models.utils as utils



def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    #MLP
    
    INPUT_DIM = (image_size,image_size, 1)
    OUTPUT_CLASSES = len(pd.unique(data[class_column]))
    HIDDEN_LAYER_DIM = 256

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=INPUT_DIM),
        tf.keras.layers.Dense(HIDDEN_LAYER_DIM, activation='sigmoid'),
        tf.keras.layers.Dense(HIDDEN_LAYER_DIM, activation='sigmoid'), #Maybe I can change the hidden layer to see if i can make it more accurate
        tf.keras.layers.Dense(OUTPUT_CLASSES)
    ])

    model.compile(
        optimizer='adam', #this used to be SGD and idk if i can just change it to adam without changing other stuff but the results look better
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = [
            utils.f1_metric(OUTPUT_CLASSES),
            utils.f1_metric_per_class(OUTPUT_CLASSES),
            "categorical_accuracy"])
    
    history = model.fit(train_generator, validation_data = val_generator, epochs=epochs, verbose=0)
    
    return model, history.history
