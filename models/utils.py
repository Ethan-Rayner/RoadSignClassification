import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prep_data(file, split):
    data = pd.read_csv(file)

    train_data, val_data = train_test_split(
        data, 
        test_size = split, 
        random_state = 0)
    
    print(
        "Test set: {} rows\nValidation set: {} rows"
        .format(train_data.shape[0], val_data.shape[0]))
    
    return data, train_data, val_data

def create_generators(train_data, val_data, image_size, class_column, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range = 15, 
        width_shift_range = 0.2,
        height_shift_range = 0.2, 
        brightness_range = [0.5, 1.5])
        
    val_datagen = ImageDataGenerator(
        rotation_range = 15, 
        width_shift_range = 0.2,
        height_shift_range = 0.2, 
        brightness_range = [0.5, 1.5])

    # categorical mode only works with one-hot encoding?
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_data,
        directory = "./",
        x_col = "path",
        y_col = class_column,
        target_size = (image_size, image_size),
        batch_size = batch_size,
        class_mode = "categorical") 

    val_generator = val_datagen.flow_from_dataframe(
        dataframe = val_data,
        directory = "./",
        x_col = "path",
        y_col = class_column,
        target_size = (image_size, image_size),
        batch_size = batch_size,
        class_mode = "categorical")
    
    return train_generator, val_generator

def plot_learning_curve(history):
    plt.figure(figsize = (10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], "r--")
    plt.plot(history["val_loss"], "b--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc = "upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history["categorical_accuracy"], "r--")
    plt.plot(history["val_categorical_accuracy"], "b--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc = "upper left")

    plt.show()
