import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.f1_score import calc_f1_scores

MAX_PER_COLUMN = 5

def create_test_generator(file, class_column, image_size, batch_size, preprocessor = None):
    data = pd.read_csv(file)

    # Create new image data generator object.
    generator = ImageDataGenerator(
        # Rescaling converts the RGB values to [0, 1] rather than [0, 255]
        rescale = 1.0 / 255,
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

def show_visual_results(model, images, num_to_test):
    class_names = []
    for class_name in images.class_indices:
        class_names.append(class_name)
    
    results_cols = MAX_PER_COLUMN
    results_rows = math.ceil(num_to_test / MAX_PER_COLUMN)
    plt.figure(figsize = (results_cols * 2, results_rows * 2))

    i = 0
    for x, y in images:
        i += 1
        if i > num_to_test:
            break

        y_prediction = model.predict(x, verbose = 0)

        x = np.squeeze(x)
        plt.subplot(results_rows, results_cols, i)
        plt.imshow(x, cmap='gray')

        actual_class = np.argmax(y[0])
        predicted_class = np.argmax(y_prediction[0])
        predicted_class_name = class_names[predicted_class]
        
        if actual_class == predicted_class:
            plt.title("{} ✓".format(predicted_class_name))
        else:
            plt.title("{} X".format(predicted_class_name))

        plt.axis("off")


def score_accuracy(model, images, num_to_test):
    # todo?: just use
    # model.evaluate(test_split_images, test_labels_split) as per
    # https://www.kaggle.com/code/tusharsharma118/belgian-traffic-dataset/notebook [34]

    class_names = []
    for class_name in images.class_indices:
        class_names.append(class_name)
    
    i = 0
    score = 0
    for x, y in images:
        i += 1
        if i > num_to_test:
            break

        y_prediction = model.predict(x, verbose=0)

        actual_class = np.argmax(y[0])
        predicted_class = np.argmax(y_prediction[0])
        if actual_class == predicted_class:
            score += 1

    
    return score / num_to_test

def score_f1(model, images, num_to_test):
    class_names = []
    for class_name in images.class_indices:
        class_names.append(class_name)
    
    i = 0
    actuals = []
    preds = []
    for x, y in images:
        i += 1
        if i > num_to_test:
            break

        y_prediction = model.predict(x, verbose=0)

        actual_class = np.argmax(y[0])
        predicted_class = np.argmax(y_prediction[0])
        actuals.append(actual_class)
        preds.append(predicted_class)
    
    f1_scores = calc_f1_scores(actuals, preds, len(class_names))
    return { class_names[i]: f1_scores[i] for i in range(len(class_names)) }