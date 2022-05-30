import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prep_data(file, split):
    data = pd.read_csv(file)

    # Do holdout validation on the dataset, then print the number of rows for
    # each dataframe.
    train_data, val_data = train_test_split(
        data, 
        test_size = split, 
        random_state = 0)
    print("Training set: {} rows\nValidation set: {} rows"
        .format(train_data.shape[0], val_data.shape[0]))
    
    return data, train_data, val_data

def create_generator(data, class_column, image_size, batch_size, preprocessor = None):
    # Create new image data generator object. This generates images to feed to 
    # the ML model, by taking the raw images and randomly applying rotation, 
    # translations, and brightness adjustments.
    generator = ImageDataGenerator(
        # Rescaling converts the RGB values to [0, 1] rather than [0, 255]
        rescale = 1.0 / 255,
        
        zoom_range = 0.2,
        rotation_range = 15, 
        width_shift_range = 0.2,
        height_shift_range = 0.2, 
        brightness_range = [0.5, 1.5],
        
        preprocessing_function = preprocessor)

    # Feed the generator the raw images from the paths provided in the "path"
    # column of the dataframe.
    iterator = generator.flow_from_dataframe(
        dataframe = data,
        directory = "./",
        x_col = "path",
        y_col = class_column,
        target_size = (image_size, image_size),
        batch_size = batch_size,
        color_mode="grayscale")

    return iterator

def f1_metric(num_of_classes):
    return tfa.metrics.F1Score(
        num_classes = num_of_classes,
        average = "macro")

def f1_metric_per_class(num_of_classes):
    return tfa.metrics.F1Score(
        num_classes = num_of_classes,
        average = None,
        name = "f1_score_per_class")

def history_graph(history):
    plt.figure(figsize = (12, 4))
    
    # The first graph is a graph of Loss vs Epochs
    plt.subplot(1, 3, 1)
    plt.plot(history["loss"], "r--")
    plt.plot(history["val_loss"], "b--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc = "upper left")

    # The second graph is a graph of F1 Score vs Epochs
    plt.subplot(1, 3, 2)
    plt.plot(history["f1_score"], "r--")
    plt.plot(history["val_f1_score"], "b--")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend(["Train", "Validation"], loc = "upper left")

    # The third graph is a graph of Accuracy vs Epochs
    plt.subplot(1, 3, 3)
    plt.plot(history["categorical_accuracy"], "r--")
    plt.plot(history["val_categorical_accuracy"], "b--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc = "upper left")

    plt.show()
