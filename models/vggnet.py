import pandas as pd
import tensorflow as tf
import models.nnet as nnet
import models.utils as utils

def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    # Calculate how many different classes of image there are.
    num_of_classes = len(pd.unique(data[class_column]))

    # A modified version of the VGGNet architecture. The original VGGNet network
    # used 16 convolutional layers to support images that were 224x224. As our
    # images are only 28x28, fewer convolutional layers will be used.
    model = tf.keras.Sequential([
        nnet.input_layer(image_size),
        nnet.conv_layer(size = image_size, filter = 3, stride = 1),
        nnet.pooling_layer(filter = 2, stride = 2),
        
        nnet.conv_layer(size = image_size * 2, filter = 3, stride = 1),
        nnet.conv_layer(size = image_size * 2, filter = 3, stride = 1),
        nnet.pooling_layer(filter = 2, stride = 2),
        
        nnet.conv_layer(size = image_size * 4, filter = 3, stride = 1),
        nnet.conv_layer(size = image_size * 4, filter = 3, stride = 1),
        nnet.pooling_layer(filter = 2, stride = 2),
        
        nnet.flatten_layer(),
        nnet.dense_layer(size = image_size * 4),
        nnet.output_layer(num_of_classes),
    ])

    # Initialize computation graph.
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [
            utils.f1_metric(num_of_classes), 
            utils.f1_metric_per_class(num_of_classes),
            "categorical_accuracy"])

    # Fit the model to the data.    
    history = model.fit(
        train_generator,
        validation_data = val_generator, 
        epochs = epochs, 
        verbose = 0)
    
    return model, history.history