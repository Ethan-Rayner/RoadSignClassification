import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def score_f1(model, images, test_size = 128):
    results = model.evaluate(images, verbose = 0, batch_size = test_size)
    print("Overall F1 Macro Score: {:.4f}".format(results[1]))
    
    class_names = []
    for class_name in images.class_indices:
        class_names.append(class_name)
    
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([i for i in range(len(class_names))])
    ax.set_xticklabels(labels = class_names, rotation = 90)

    ax.set_ylabel("F1 Scores")
    scores = [results[2][i] for i in range(len(class_names))]
    ax.bar(class_names, scores, color="#0055ff")

    plt.show()
