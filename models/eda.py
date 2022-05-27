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

    legend_colors = {'Training':'#0055ff', 'Validation':'#008800'}         
    legend_labels = list(legend_colors.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color = legend_colors[label]) for label in legend_labels]
    plt.legend(legend_handles, legend_labels)
    
    plt.show()