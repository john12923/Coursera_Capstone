# Auxiliary functions

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import Markdown, display
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ----------------------------------------------------------------------------
# Plots and creates labels for values plotted. Used for exploration of features.
def plot_and_annotate(data, **kwarg):
    plt.figure(figsize=(10,8))
    ax = data['percent'].sort_values().plot(kind='barh', **kwarg)
    for p in ax.patches:
        width = p.get_width()
        plt.text(width, p.get_y() + 0.55 * p.get_height(),
                 '{:1.2f}%'.format(width))
                 #ha='center', va='center')

# ----------------------------------------------------------------------------
# Creates percentages for value_counts of a pd.series object.
def create_count_percentages(series, name=None):
    values = series.value_counts(normalize=True).mul(100)
    expanded_value = values.to_frame(name)
    expanded_value['percent'] = values
    return expanded_value


# ----------------------------------------------------------------------------
# Data must be loaded before definin create_plot_data due to inside reference.
def create_plot_data(data, feature):
    return create_count_percentages(data[feature], name=feature)


#-----------------------------------------------------------------------------
def create_grp_data(data, feature):
    order = list(data[feature].unique())
    grouped = data.groupby('SEVERITYCODE')[feature].value_counts(normalize=True).mul(100)
    grouped = grouped.sort_index()
    grouped = grouped.rename('percent').reset_index()
    return order, feature, grouped


def plot_grp_data(data, categorical):
    fig, axes =plt.subplots(figsize=(15,3),nrows=1, ncols=4)
    c_list = ['P', 'M', 'S', 'F']
    colors = ['steelblue','mediumseagreen','coral','rosybrown']
    for i, ax in enumerate(axes.flat):
        if(categorical):
            sns.barplot(x='percent', y=data[1], ax=ax,data=data[2][data[2]['SEVERITYCODE']==c_list[i]], color=colors[i], order=data[0])
        else:
            sns.barplot(y='percent', x=data[1], ax=ax,data=data[2][data[2]['SEVERITYCODE']==c_list[i]], color=colors[i])
        #ax.set_xticklabels(ax.get_xticklabels())
        #ax.set_ylabel( " %")
        ax.set_title("Class "+ c_list[i])
        plt.tight_layout()
# ----------------------------------------------------------------------------


#Function to Plot Confusion Matrix
def plot_cm(y_test, y_pred):
    target_names = ['P', 'M', 'S', 'F']
    y_label = np.unique(y_test)
    accident_confusion_matrix = confusion_matrix(y_test, y_pred, labels=['P', 'M', 'S', 'F'])
    title = 'Accident Severity Confusion Matrix'
    cmap = plt.cm.Blues
    
    plt.style.use('classic')
    plt.figure(figsize=(6, 5))
    accident_confusion_matrix = (accident_confusion_matrix.astype('float') / accident_confusion_matrix.sum(axis=1)[:, np.newaxis]) * 100

    plt.imshow(accident_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_label))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)

    fmt = '.2f'
    thresh = accident_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(accident_confusion_matrix.shape[0]), range(accident_confusion_matrix.shape[1])):
        plt.text(j, i, format(accident_confusion_matrix[i, j], fmt),horizontalalignment="center",
                 color="white" if accident_confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()