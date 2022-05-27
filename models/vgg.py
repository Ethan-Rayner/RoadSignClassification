import pandas as pd
import tensorflow as tf

def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    # Calculate how many different classes of image there are.
    num_of_classes = len(pd.unique(data[class_column]))

    # Create a VGG 3x3 model, which has three blocks of convolution & pooling 
    # layers.
    model = tf.keras.Sequential([
        input_layer(image_size),
        conv_layer(image_size),
        pooling_layer(),
        
        conv_layer(image_size * 2),
        conv_layer(image_size * 2),
        pooling_layer(),
        
        conv_layer(image_size * 4),
        conv_layer(image_size * 4),
        pooling_layer(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(image_size * 4, activation = "relu"),
        tf.keras.layers.Dense(num_of_classes),
    ])

    # Initialize computation graph.
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
        metrics = ["categorical_accuracy"])

    # todo: use "sparse_categorical_crossentropy" like
    # https://www.kaggle.com/code/tusharsharma118/belgian-traffic-dataset/notebook [28]

    # Fit the model to the data.    
    history = model.fit(
        train_generator,
        validation_data = val_generator, 
        epochs = epochs, 
        verbose = 0)
    
    return model, history.history

def input_layer(image_size):
    # The first layer in the VGG model. Takes input from the image itself.
    return tf.keras.layers.Conv2D(
        image_size, 
        (3, 3), 
        activation = "relu", 
        padding = "same", 

        # Is this 3 for RGB? Because our images are only B+W...
        input_shape = (image_size, image_size, 3))

def conv_layer(size):
    # A convolution layer in the VGG model.
    return tf.keras.layers.Conv2D(
        size, 
        (3, 3), 
        activation = "relu", 
        padding = "same")

def pooling_layer():
    # A pooling layer in the VGG model.
    return tf.keras.layers.MaxPooling2D((2, 2))