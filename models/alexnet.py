import pandas as pd
import tensorflow as tf
import models.nnet as nnet
import models.utils as utils

def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    # Calculate how many different classes of image there are.
    num_of_classes = len(pd.unique(data[class_column]))

    # Alexnet
    model = tf.keras.Sequential([
        nnet.input_layer(image_size),
        nnet.conv_layer(size = 96, filter = 11, stride = 4),
        nnet.pooling_layer(filter = 3, stride = 2),
        
        nnet.padding_layer(padding = 2),
        nnet.conv_layer(size = 256, filter = 5, stride = 1),
        nnet.pooling_layer(filter = 3, stride = 2),
        
        nnet.padding_layer(padding = 1),
        nnet.conv_layer(size = 384, filter = 3, stride = 1),
        nnet.padding_layer(padding = 1),
        nnet.conv_layer(size = 384, filter = 3, stride = 1),
        nnet.padding_layer(padding = 1),
        nnet.conv_layer(size = 256, filter = 3, stride = 1),
        nnet.pooling_layer(filter = 3, stride = 2),

        nnet.flatten_layer(),
        nnet.dropout_layer(dropout = 0.5),
        nnet.output_layer(num_of_classes),
    ])

    # Initialize computation graph.
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
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
