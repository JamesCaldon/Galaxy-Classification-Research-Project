
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
RANDOM_SEED_ = 2021
rng = np.random.default_rng(RANDOM_SEED_)
def plot_classification_results(images, display_size=(5, 5), y_preds=None, y_trues=None, y_labels=None, galaxy_names=None, random_sample=False):
    display_count = np.prod(display_size)
    fig, axs = plt.subplots(display_size[0], display_size[1])
    plt.tight_layout()
    def plot_image(img, y_pred, y_true, galaxy_name, ax):
        ax.imshow(img)
        title_color = 'black'
        title = ""
        if (galaxy_name is not None):
            title += galaxy_name + str(': ')
        
        if (y_pred is not None):
            
            if (y_pred == 1):
                title += y_labels[1]
            else:
                title += y_labels[0]

        if (y_true is not None):
            if (y_pred != y_true):
                title_color = 'red'
        ax.set_title(title, color=title_color)

    if (display_count != 1):
        axs_list = axs.flatten()
    else:
        axs_list = [axs]
    selected_images = slice(0, display_count)
    if random_sample:
        selected_images = rng.permutation(images.shape[0])[:display_count]
        #selected_images = rng.integers(0, images.shape[0], display_count)

    if (galaxy_names is None):
        galaxy_names = np.full(images.shape[0], None)
        
    if (y_trues is None):
        y_trues = np.full(images.shape[0], None)
    list(map(plot_image, images[selected_images], y_preds[selected_images], y_trues[selected_images], galaxy_names[selected_images], axs_list));



from utility.data_loading import Classes

RANDOM_SEED_ = 2021
rng = np.random.default_rng(RANDOM_SEED_)
def plot_classification_results_new(images, metadata=None, title="", display_size=(5, 5), random_sample=True, predicted_classes = None):
    mpl.rcParams['figure.figsize'] = [40, 40]
    mpl.rcParams['figure.dpi'] = 72
    display_count = np.prod(display_size)
    fig, axs = plt.subplots(display_size[0], display_size[1])
    fig.suptitle(title)
    if random_sample:
        selected_images = rng.permutation(images.shape[0])[:display_count]
    else:
        selected_images = list(range(images.shape[0]))[:display_count]
        #print(selected_images)

    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].axis('off')
        if i >= len(images): continue
        axs[i].imshow(images[selected_images[i], :, :])
        if (metadata is not None):
            if ('name' in metadata.columns):
                axs[i].set_title(metadata.iloc[i]['name'].strip())
            if ('class' in metadata.columns):
                true_class = None
                if type(metadata.iloc[i]['class']) is np.int:
                    true_class = Classes(metadata.iloc[i]['class']).name
                else:
                    true_class = metadata.iloc[i]['class']
                ant = f"Class: {true_class}"
                if predicted_classes is not None:
                    ant = "True " + ant
                    anp = f"Predicted Class: {Classes(predicted_classes[i]).name}"
                    pcolor = 'white'
                    if Classes(predicted_classes[i]).name != true_class:
                        pcolor = 'red'
                    axs[i].annotate(anp,
                        xy=(0.05, 0.15),
                        xycoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=10,
                        color=pcolor
                    )
                axs[i].annotate(ant,
                    xy=(0.05, 0.25),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=10,
                    color='white'
                )
    fig.tight_layout()