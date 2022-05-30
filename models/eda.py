import matplotlib.pyplot as plt
from matplotlib import gridspec

def classes_histogram(train_data, val_data, class_cols):
    spec = gridspec.GridSpec(
        ncols = 2, 
        nrows = 1,
        width_ratios = [1, 3])

    plt.figure(figsize = (12, 5))
    for i in range(len(class_cols)):
        class_column = class_cols[i]
        plt.subplot(spec[i])
        
        class_names = list(train_data[class_column].unique())
        plt.ylabel("Images in dataset")
        plt.xticks([i for i in range(len(class_names))], labels = class_names, rotation = 90)

        train_data_freqs = [train_data.loc[train_data[class_column] == i].shape[0] for i in class_names]
        plt.bar(class_names, train_data_freqs, color="#0055ff")

        val_data_freqs = [val_data.loc[val_data[class_column] == i].shape[0] for i in class_names]
        plt.bar(class_names, val_data_freqs, color="#008800")

        legend_colors = {'Training':'#0055ff', 'Validation':'#008800'}         
        legend_labels = list(legend_colors.keys())
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color = legend_colors[label]) for label in legend_labels]
        plt.legend(legend_handles, legend_labels)
    
    plt.show()