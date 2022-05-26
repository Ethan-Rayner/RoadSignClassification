import pandas as pd
import tensorflow as tf

def fit_model(data, class_column, train_generator, val_generator, image_size, epochs):
    # Calculate how many different classes of image there are.
    num_of_classes = len(pd.unique(data[class_column]))

    # Create a Resnet model.
    model = tf.keras.applications.resnet50.ResNet50(
        weights = None,
        classes = num_of_classes,
        classifier_activation = None
    )

    # Initialize computation graph.
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
        metrics = ["categorical_accuracy"])

    # Fit the model to the data.    
    history = model.fit(
        train_generator,
        validation_data = val_generator, 
        epochs = epochs, 
        verbose = 0)
    
    return model, history.history