import pandas as pd
from utility.data_loading import Classes
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import numpy as np

def write_phys_prop_table(metadata, pred, properties, model_name):
    c2_NA10_E = metadata.iloc[pred == Classes.E]
    c2_NA10_ES = metadata.iloc[pred == Classes.ES]

    df_C2_means = pd.DataFrame([c2_NA10_E[properties].mean(), c2_NA10_ES[properties].mean(), pd.Series.subtract(c2_NA10_E[properties].mean(), c2_NA10_ES[properties].mean())], index=["E Mean", "ES Mean", "Difference"]).T
    df_C2_means.insert(0, "Physical Property", df_C2_means.index)
    df_C2_means.style.set_caption(f"{model_name}: Differences in Physical Properties between Predicted E and ES Classes")
    display(df_C2_means.to_markdown(index=False))
    latex = df_C2_means.to_latex(index=False, escape=True, caption=f"{model_name}: Differences in Physical Properties between Predicted E and ES Classes", float_format="%0.2f")
    latex = latex.replace(r"\toprule", r"\hline")
    latex = latex.replace(r"\midrule", r"\hline")
    latex = latex.replace(r"\bottomrule", r"\hline")

    with open(f"E:\\OneDrive - The University of Western Australia\\Documents\\Honours - Galaxy Classification\\Papers\\Dissertation\\physprop_{model_name}.tex", 'w') as f:
        f.write(latex)
        f.flush()

def plot_phys_prop_hists(metadata, pred, properties, model_name):

    colours = ["blue", "skyblue"]
    preds = [metadata.iloc[pred == Classes.E], metadata.iloc[pred == Classes.ES]]
    classes = ["E", "ES"]
    fig, axs = plt.subplots(2, 4)#, figsize=(100, 100))
    fig.suptitle(f"{model_name}: Predicted E and ES Physical Properties")
    plt.tight_layout()
    for property, ax in zip(properties, axs.flatten()):
        legend = []
        for pred, colour, cls in zip(preds, colours, classes):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            na_begone = pred[property].dropna()
            w = (np.ones(len(na_begone))/len(na_begone))
            na_begone.hist(bins=6, legend=True, ax=ax, color = colour, alpha=0.75, weights = w)
            legend.append(f"{cls}: {property}")
        ax.legend(legend, fontsize='large')

def plot_class_distributions(pred, model_name):
    results = pd.DataFrame({"Class": ["E", "ES"], 
    "Count": [np.count_nonzero(pred == Classes.E), np.count_nonzero(pred == Classes.ES)]})
    splot=sns.barplot(x="Class", y="Count", data=results)
    plt.bar_label(splot.containers[0], label_type="center")

    total = len(pred)
    for p in splot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.1
        y = p.get_y() + p.get_height() + 10
        splot.annotate(percentage, (x, y), size = 12)
        splot.set_title(f"{model_name}: Class Distributions")
    plt.show()


import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
RANDOM_SEED_ = 2021
rng = np.random.default_rng(RANDOM_SEED_)
def plot_dataset(images, metadata=None, title="", display_size=(5, 5), random_sample=True):
    display_count = np.prod(display_size)
    fig, axs = plt.subplots(display_size[0], display_size[1])
    fig.suptitle(title, fontsize='xx-large')
    if random_sample:
        selected_images = rng.permutation(images.shape[0])[:display_count]
        #print(selected_images[:5])
    name_field = 'name'
    
    if (metadata is not None and 'JID' in metadata.columns):
        name_field = 'JID'
    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].axis('off')
        axs[i].imshow(images[selected_images[i], :, :])
        if (metadata is not None):
            
            if (name_field in metadata.columns):
                if (metadata[name_field].iloc[0] != 'sim'):
                    axs[i].set_title(metadata.iloc[selected_images[i]][name_field].strip())

                if ('class' in metadata.columns and name_field != 'JID'):
                    #print(metadata.iloc[selected_images[i]]['class'])
                    if type(metadata.iloc[selected_images[i]]['class']) is np.int:
                        an = f"Class: {Classes(metadata.iloc[selected_images[i]]['class']).name}"
                    else:
                        an = f"Class: {metadata.iloc[selected_images[i]]['class']}"
                    axs[i].annotate(an,
                        xy=(0.55, 0.15),
                        xycoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=10,
                        color='white'
                    )

    fig.tight_layout()
        

def plot_datagen(gen, images, **kwargs):
    count = (5, 5)
    if 'display_size' in kwargs:
        count = kwargs['display_size']
    plot_dataset(images, images, **kwargs)

