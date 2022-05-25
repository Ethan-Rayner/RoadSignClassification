import pandas as pd
import tensorflow as tf

def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    num_of_classes = len(pd.unique(data[class_column]))

    model = tf.keras.Sequential([
        # VGG block 1
        input_layer(image_size),
        conv_layer(image_size),
        pooling_layer(),
        
        # VGG block 2
        conv_layer(image_size * 2),
        conv_layer(image_size * 2),
        pooling_layer(),
        
        # VGG block 3
        conv_layer(image_size * 4),
        conv_layer(image_size * 4),
        pooling_layer(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(image_size * 4, activation = "relu"),
        tf.keras.layers.Dense(num_of_classes),
    ])

    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
        metrics = ["categorical_accuracy"])
    
    history = model.fit(
        train_generator,
        validation_data = val_generator, 
        epochs = epochs, 
        verbose = 0)
    
    return history.history

def input_layer(image_size):
    return tf.keras.layers.Conv2D(
        image_size, 
        (3, 3), 
        activation = "relu", 
        padding = "same", 

        # Is this 3 for RGB? Because our images are only B+W...
        input_shape = (image_size, image_size, 3))

def conv_layer(size):
    return tf.keras.layers.Conv2D(
        size, 
        (3, 3), 
        activation = "relu", 
        padding = "same")

def pooling_layer():
    return tf.keras.layers.MaxPooling2D((2, 2))