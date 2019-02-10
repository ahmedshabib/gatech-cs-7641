import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12


# Phishing dataset requires some preprocessing
# We cleanup the dataset to have the features with values 0 or 1
def get_phishing_data():
    df_phish = pd.read_csv('data/phishing-website.csv').astype('category')

    col_1hot = ['URL_Length', 'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
                'web_traffic', 'Links_pointing_to_page']
    df_1hot = pd.get_dummies(df_phish[col_1hot])
    df_others = df_phish.drop(col_1hot, axis=1)
    df_phish = pd.concat([df_1hot, df_others], axis=1)
    df_phish = df_phish.replace(-1, 0).astype('category')
    column_order = list(df_phish)
    column_order.insert(0, column_order.pop(column_order.index('Result')))
    df_phish = df_phish.loc[:, column_order]  # move the target variable 'Result' to the front
    X = np.array(df_phish.values[:, 1:-1], dtype='int64')
    Y = np.array(df_phish.values[:, 0], dtype='int64')
    return X, Y


# Vocal Data just requires minor preprocessing, the output is 1,2 , we just need to change it to binary range(0,1)
def get_vocal_data():
    df_vocal = pd.read_csv('data/nasal-oral.csv').astype('category')
    X = np.array(df_vocal.values[:, 1:-1])
    # Y = np.array(df_vocal.values[:, 0], dtype='int64')
    df_1hot = df_vocal['Class']
    df_1hot = pd.get_dummies(df_1hot)
    Y = df_1hot.values[:, 0]
    return X, Y


def plot_learning_curve(clf, X, y, title="Insert Title"):
    n = len(y)
    train_mean = [];
    train_std = []  # model performance score (f1)
    cv_mean = [];
    cv_std = []  # model performance score (f1)
    fit_mean = [];
    fit_std = []  # model fit/training time
    pred_mean = [];
    pred_std = []  # model test/prediction times
    train_sizes = (np.linspace(.05, 1.0, 20) * n).astype('int')

    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx, :]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)

        train_mean.append(np.mean(scores['train_score']));
        train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score']));
        cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time']));
        fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time']));
        pred_std.append(np.std(scores['score_time']))

    train_mean = np.array(train_mean);
    train_std = np.array(train_std)
    cv_mean = np.array(cv_mean);
    cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean);
    fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean);
    pred_std = np.array(pred_std)

    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)

    return train_sizes, train_mean, fit_mean, pred_mean


def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    plt.figure()
    plt.title("Learning Curve: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2 * train_std, train_mean + 2 * train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()


def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    plt.figure()
    plt.title("Modeling Time: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2 * fit_std, fit_mean + 2 * fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def final_classifier_evaluation(clf, X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    start_time = timeit.default_timer()
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()
