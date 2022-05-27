import matplotlib.pyplot as plt

def classes_histogram(train_data, val_data, class_column):
    print()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    classes = list(train_data[class_column].unique())
    ax.set_xticks([i for i in range(len(classes))])
    ax.set_xticklabels(labels = classes, rotation = 90)

    train_data_freqs = [train_data.loc[train_data[class_column] == i].shape[0] for i in classes]
    ax.bar(classes, train_data_freqs, color="#0055ff")

    val_data_freqs = [val_data.loc[val_data[class_column] == i].shape[0] for i in classes]
    ax.bar(classes, val_data_freqs, color="#008800")
    
    plt.show()