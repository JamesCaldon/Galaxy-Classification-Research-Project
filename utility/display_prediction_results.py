
def evaluate(model, datagen, x_dataset, y_dataset, datagen_batch_size=32):
    model.evaluate(datagen.flow(x_dataset, y_dataset, batch_size=datagen_batch_size, subset='validation'), verbose=0)

def plot_learning_curves(train_history, metrics = ['loss', 'sparse_categorical_accuracy', 'binary_accuracy', 'auc']):
    from matplotlib import pyplot as plt
    for metric in metrics:
        if (metric in train_history.history.keys()):
            plt.plot(train_history.history[metric])
            plt.plot(train_history.history[f"val_{metric}"])
            plt.legend([metric, f"val_{metric}"])
            plt.show()
    
def plot_model_results(model, train_data, train_labels, val_data, val_labels, total_data, total_labels, model_name=""):
    print("Training Results: ")
    plot_testing_results(model.predict(train_data).argmax(axis=1), train_labels, f"{model_name} Training")
    print("Validation Results: ")
    plot_testing_results(model.predict(val_data).argmax(axis=1), val_labels, f"{model_name} Validation")
    print("Combined Results: ")
    plot_testing_results(model.predict(total_data).argmax(axis=1), total_labels, f"{model_name}")

def get_results(model, data, labels):

    true = labels # maybe labels or true should be renamed?
    pred = model.predict(data)

    if model.layers[-1].output.shape[-1] > 1:
        pred = pred.argmax(axis=1)
    else:
        pred = pred.round()
    
    from sklearn import metrics
    results = {
        "Accuracy": metrics.accuracy_score(true, pred),
        "Precision": metrics.precision_score(true, pred),
        "Recall": metrics.recall_score(true, pred),
        "F1": metrics.f1_score(true, pred),
        "AUC": metrics.roc_auc_score(true, pred),
        "ConfusionMatrix": metrics.confusion_matrix(true, pred)
    }
    return results

def plot_results(results, average = True, title="", subplot_titles=[]):
    if (isinstance(results, dict)):
        all_results = [results]
    else:
        all_results = results
    if (average):
        average_results = {}
        for result in all_results:
            for key in result:
                if (key not in average_results):
                    average_results.update({key: result[key]})
                else:
                    average_results.update({key: average_results[key] + result[key]})
        average_results = {k: v / len(all_results) for k, v in average_results.items()}
        make_confusion_matrix(np.array(average_results['ConfusionMatrix']), categories=["E", "ES"], cmap='Blues', title=title)
    else:
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        fig.suptitle(title)
        for result, title, ax in zip(all_results, subplot_titles, axs):
            make_confusion_matrix(np.array(result['ConfusionMatrix']), categories=["E", "ES"], cmap='Blues', title=title, ax=ax)
        fig.set_constrained_layout(True)



def plot_model_results_from_datagen(model, val_datagen, val_data, val_labels):
    print("Validation Results: ")
    print(model.evaluate(val_datagen.flow(val_data, val_labels)))


def plot_testing_results(pred, true, model_name=""):
    from sklearn import metrics
    print("Accuracy: ", str(metrics.accuracy_score(true, pred)))
    print("Precision: ", str(metrics.precision_score(true, pred)))
    print("Recall: ", str(metrics.recall_score(true, pred)))
    print("F1 Score: ", str(metrics.f1_score(true, pred)))
    print("AUC: ", str(metrics.roc_auc_score(true, pred)))

    cmd = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(true, pred), display_labels=["E", "ES"])
    cmd = cmd.plot(cmap='Blues')
    cmd.ax_.set_title(f"{model_name} Confusion Matrix")


# Taken and modified with permission from DTrimarchi10's Github Repo https://github.com/DTrimarchi10/confusion_matrix:
# see: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          ax=None,
                          true_pred_labels = ['True label', 'Predicted label']):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    ax:            Plot on provided matplotlib axes, otherwise create new one. Title is ignored if this is set.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        #group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        group_percentages = ["{0:.2%}".format(value) for value in (cf.T/np.sum(cf, axis=1)).T.flatten()] # Modification to present percentage in terms of row

    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[0,0] / sum(cf[:,0])
            recall    = cf[0,0] / sum(cf[0,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    if (ax is None):
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            fig.suptitle(title)
    else:
        ax.set_title(title)

    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, ax=ax)
    if xyplotlabels:
        ax.set_ylabel(true_pred_labels[0])
        ax.set_xlabel(true_pred_labels[1] + stats_text)
    else:
        ax.set_xlabel(stats_text)