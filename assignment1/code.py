# Datasets
# 1. https://www.openml.org/d/1489
# 2. https://www.openml.org/d/4534
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from assignment1.helper import get_phishing_data, get_vocal_data, final_classifier_evaluation, plot_learning_curve
from sklearn.tree import DecisionTreeClassifier

# X_p, Y_p = get_phishing_data()
# X_v, Y_v = get_vocal_data()

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12


# Decision_tree
def hyperTree(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    max_depth = list(range(1, 25))
    for i in max_depth:
        clf = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(max_depth, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(max_depth, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Max Tree Depth')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def hyperTreeSampleLeaf(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))
    min_samples_leaf = np.linspace(start_leaf_n, end_leaf_n, 20).round().astype('int')
    for i in min_samples_leaf:
        clf = DecisionTreeClassifier(random_state=100, min_samples_leaf=i, criterion='entropy')
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(min_samples_leaf, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(min_samples_leaf, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Min Sample Leaf Depth')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def TreeGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    param_grid = {
        'min_samples_leaf': np.linspace(start_leaf_n, end_leaf_n, 20).round().astype('int'),
        'max_depth': np.arange(1, 20)
    }

    tree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=10)
    tree.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(tree.best_params_)
    return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']


def generate_decision_tree_analysis():
    X_p, Y_p = get_phishing_data()
    X_v, Y_v = get_vocal_data()
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperTree(X_train, y_train, X_test, y_test,
              title="Model Complexity Curve for Decision Tree (Phishing Data)\nHyperparameter : Tree Max Depth")
    hyperTreeSampleLeaf(X_train, y_train, X_test, y_test,
                        title="Model Complexity Curve for Decision Tree (Phishing Data)\nHyperparameter : Min Sample Leaf")
    max_depth, min_samples_leaf = TreeGridSearchCV(round(0.005 * len(X_train)), round(0.05 * len(X_train)), X_train,
                                                   y_train)
    estimator_phish = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100,
                                             criterion='entropy')
    train_samp_phish, DT_train_score_phish, DT_fit_time_phish, DT_pred_time_phish = plot_learning_curve(estimator_phish,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        title="Decision Tree Phishing Data")
    final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_v), np.array(Y_v), test_size=0.20)
    hyperTree(X_train, y_train, X_test, y_test,
              title="Model Complexity Curve for Decision Tree (Vocal Data)\nHyperparameter : Tree Max Depth")
    hyperTreeSampleLeaf(X_train, y_train, X_test, y_test,
                        title="Model Complexity Curve for Decision Tree (Vocal Data)\nHyperparameter : Min Sample Leaf")
    max_depth, min_samples_leaf = TreeGridSearchCV(round(0.005 * len(X_train)), round(0.05 * len(X_train)), X_train,
                                                   y_train)
    estimator_vocal = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100,
                                             criterion='entropy')
    train_samp_vocal, DT_train_score_vocal, DT_fit_time_vocal, DT_pred_time_vocal = plot_learning_curve(estimator_vocal,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        title="Decision Tree Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


# Neural Networks
def hyperNN(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    hlist = np.linspace(1, 100, 20).astype('int')
    for i in hlist:
        clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic',
                            learning_rate_init=0.05, random_state=100)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(hlist, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Hidden Units')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def hyperNNLR(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    lr = [0.01, 0.025, 0.05, 0.075, .1]
    for i in lr:
        clf = MLPClassifier(solver='adam', activation='logistic',
                            learning_rate_init=i, random_state=100)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(lr, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(lr, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Learning Rate')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def NNGridSearchCV(X_train, y_train):
    lr = [0.01, 0.05, .1]
    hidden_layer_sizes = [5, 10, 20, 30, 40, 50, 75, 100]
    param_grid = {'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate_init': lr}

    net = GridSearchCV(estimator=MLPClassifier(solver='adam', activation='logistic', random_state=100),
                       param_grid=param_grid, cv=10)
    net.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(net.best_params_)
    return net.best_params_['hidden_layer_sizes'], net.best_params_['learning_rate_init']


def generate_neural_network_analysis():
    X_p, Y_p = get_phishing_data()
    X_v, Y_v = get_vocal_data()

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperNN(X_train, y_train, X_test, y_test,
            title="Model Complexity Curve for NN (Phishing Data)\nHyperparameter : No. Hidden Units")
    hyperNNLR(X_train, y_train, X_test, y_test,
              title="Model Complexity Curve for NN (Phishing Data)\nHyperparameter : Learning Rate")
    h_units, learn_rate = NNGridSearchCV(X_train, y_train)
    estimator_phish = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic',
                                    learning_rate_init=learn_rate, random_state=100)
    train_samp_phish, NN_train_score_phish, NN_fit_time_phish, NN_pred_time_phish = plot_learning_curve(estimator_phish,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        title="Neural Net Phishing Data")
    final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_v), np.array(Y_v), test_size=0.20)
    hyperNN(X_train, y_train, X_test, y_test,
            title="Model Complexity Curve for NN (Vocal Data)\nHyperparameter : No. Hidden Units")
    hyperNNLR(X_train, y_train, X_test, y_test,
              title="Model Complexity Curve for NN (Vocal Data)\nHyperparameter : Learning Rate")
    h_units, learn_rate = NNGridSearchCV(X_train, y_train)
    estimator_vocal = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic',
                                    learning_rate_init=learn_rate, random_state=100)
    train_samp_vocal, NN_train_score_vocal, NN_fit_time_vocal, NN_pred_time_vocal = plot_learning_curve(estimator_vocal,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        title="Neural Net Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


def hyperBoost(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title):
    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1, 100, 20).astype('int')
    for i in n_estimators:
        # print(i)
        clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth / 2),
                                         min_samples_leaf=int(min_samples_leaf / 2), random_state=100, )
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(n_estimators, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def BoostedGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    # parameters to search:
    # n_estimators, learning_rate, max_depth, min_samples_leaf
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n, end_leaf_n, 3).round().astype('int'),
                  'max_depth': np.linspace(3, 24, 3).round().astype('int'),
                  'n_estimators': np.linspace(10, 100, 5).round().astype('int'),
                  'learning_rate': [0.01, 0.05, .1]}

    boost = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=10)
    boost.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(boost.best_params_)
    return boost.best_params_['max_depth'], boost.best_params_['min_samples_leaf'], boost.best_params_['n_estimators'], \
           boost.best_params_['learning_rate']


def generate_boosting_analysis():
    X_p, Y_p = get_phishing_data()
    X_v, Y_v = get_vocal_data()

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperBoost(X_train, y_train, X_test, y_test, 3, 50,
               title="Model Complexity Curve for Boosted Tree (Phishing Data)\nHyperparameter : No. Estimators")
    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))  # leaf nodes of size [0.5%, 5% will be tested]
    max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train)
    estimator_phish = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                 n_estimators=n_est, learning_rate=learn_rate, random_state=100)
    train_samp_phish, BT_train_score_phish, BT_fit_time_phish, BT_pred_time_phish = plot_learning_curve(
        estimator_phish,
        X_train,
        y_train,
        title="Boosted Tree Phishing Data")
    final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_v), np.array(Y_v), test_size=0.20)
    hyperBoost(X_train, y_train, X_test, y_test, 3, 50,
               title="Model Complexity Curve for Boosted Tree (Vocal Data)\nHyperparameter : No. Estimators")
    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))  # leaf nodes of size [0.5%, 5% will be tested]
    max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train)
    estimator_vocal = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                 n_estimators=n_est, learning_rate=learn_rate, random_state=100)
    train_samp_vocal, BT_train_score_vocal, BT_fit_time_vocal, BT_pred_time_vocal = plot_learning_curve(
        estimator_vocal,
        X_train, y_train,
        title="Boosted Tree Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


def hyperKNN(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    klist = np.arange(1, 25)
    for i in klist:
        clf = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(klist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def generate_knn_analysis():
    X_p, Y_p = get_phishing_data()
    X_v, Y_v = get_vocal_data()

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperKNN(X_train, y_train, X_test, y_test,
             title="Model Complexity Curve for kNN (Phishing Data)\nHyperparameter : No. Neighbors")
    estimator_phish = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    train_samp_phish, kNN_train_score_phish, kNN_fit_time_phish, kNN_pred_time_phish = plot_learning_curve(
        estimator_phish, X_train, y_train, title="kNN Phishing Data")
    final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_v), np.array(Y_v), test_size=0.20)
    hyperKNN(X_train, y_train, X_test, y_test,
             title="Model Complexity Curve for kNN (Vocal Data)\nHyperparameter : No. Neighbors")
    estimator_vocal = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    train_samp_vocal, kNN_train_score_vocal, kNN_fit_time_vocal, kNN_pred_time_vocal = plot_learning_curve(
        estimator_vocal,
        X_train, y_train,
        title="kNN Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


def hyperSVM(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    kernel_func = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in kernel_func:
        if i == 'poly':
            for j in [2, 3, 4, 5, 6, 7, 8]:
                clf = SVC(kernel=i, degree=j, random_state=100)
                clf.fit(X_train, y_train)
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                f1_test.append(f1_score(y_test, y_pred_test))
                f1_train.append(f1_score(y_train, y_pred_train))
        else:
            clf = SVC(kernel=i, random_state=100)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))

    xvals = ['linear', 'poly2', 'poly3', 'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'rbf', 'sigmoid']
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def hyperSVMPenalty(X_train, y_train, X_test, y_test, title):
    f1_test = []
    f1_train = []
    x_vals = [0.0001, 0.001, 0.01, 0.1, 1]
    for i in x_vals:
        clf = SVC(kernel='rbf', random_state=100, C=i)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(x_vals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(x_vals, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Penalty Parameter (C)')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def SVMGridSearchCV(X_train, y_train):
    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs}

    clf = GridSearchCV(estimator=SVC(kernel='rbf', random_state=100),
                       param_grid=param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    return clf.best_params_['C']


def generate_svm_analysis():
    X_p, Y_p = get_phishing_data()
    X_v, Y_v = get_vocal_data()

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperSVM(X_train, y_train, X_test, y_test,
             title="Model Complexity Curve for SVM (Phishing Data)\nHyperparameter : Kernel Function")
    hyperSVMPenalty(X_train, y_train, X_test, y_test,
                    title="Model Complexity Curve for SVM (Phishing Data)\nHyperparameter : Penalty (C)")
    C_val = SVMGridSearchCV(X_train, y_train)
    estimator_phish = SVC(C=C_val, kernel='rbf', random_state=100)
    train_samp_phish, SVM_train_score_phish, SVM_fit_time_phish, SVM_pred_time_phish = plot_learning_curve(
        estimator_phish, X_train, y_train, title="SVM Phishing Data")
    final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_v), np.array(Y_v), test_size=0.20)
    hyperSVM(X_train, y_train, X_test, y_test,
             title="Model Complexity Curve for SVM (Vocal Data)\nHyperparameter : Kernel Function")
    hyperSVMPenalty(X_train, y_train, X_test, y_test,
                    title="Model Complexity Curve for SVM (Vocal Data)\nHyperparameter : Penalty(C)")
    C_val = SVMGridSearchCV(X_train, y_train)
    estimator_vocal = SVC(C=C_val, kernel='rbf', random_state=100)
    train_samp_vocal, SVM_train_score_vocal, SVM_fit_time_vocal, SVM_pred_time_vocal = plot_learning_curve(
        estimator_vocal,
        X_train, y_train,
        title="SVM Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


generate_decision_tree_analysis()
generate_neural_network_analysis()
generate_boosting_analysis()
generate_svm_analysis()
generate_knn_analysis()
