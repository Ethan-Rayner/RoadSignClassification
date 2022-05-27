import tensorflow as tf

def input_layer(image_size):
    # Takes input from the image itself.
    return tf.keras.layers.InputLayer(input_shape = (image_size, image_size, 1))

def conv_layer(size, filter, stride):
    # A convolution layer with the given filter size and stride length.
    return tf.keras.layers.Conv2D(
        size, 
        (filter, filter),
        strides = (stride, stride),
        activation = "relu", 
        padding = "same")

def padding_layer(padding):
    return tf.keras.layers.ZeroPadding2D(
        padding = (padding, padding))

def pooling_layer(filter, stride):
    # A max pooling layer with the given filter size and string length.
    return tf.keras.layers.MaxPooling2D(
        (filter, filter), 
        strides = (stride, stride))

def flatten_layer():
    # Convert the 3-dimensional group of nodes into a 1 dimensional group of nodes. 
    return tf.keras.layers.Flatten()

def dropout_layer(dropout):
    # Disables a random sample of nodes each epoch based on the given percentage.
    return tf.keras.layers.Dropout(dropout)

def dense_layer(size):
    # The final layer containing the number of classes.
    return tf.keras.layers.Dense(size, activation = "relu")

def output_layer(num_of_classes):
    # The final layer containing the number of classes.
    return tf.keras.layers.Dense(num_of_classes)
