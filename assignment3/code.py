import timeit

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix, \
    homogeneity_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from assignment3.helper import get_phishing_data, get_vocal_data, plot_learning_curve, final_classifier_evaluation, \
    plot_confusion_matrix, cluster_predictions, pairwiseDistCorr, compare_fit_time, compare_pred_time, \
    compare_learn_time
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.mixture import GaussianMixture as EM
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12


#
#
# def import_data():
#     X1 = np.array(df_phish.values[:, 1:-1], dtype='int64')
#     Y1 = np.array(df_phish.values[:, 0], dtype='int64')
#     X2 = np.array(df_bank.values[:, 1:-1], dtype='int64')
#     Y2 = np.array(df_bank.values[:, 0], dtype='int64')
#     return X1, Y1, X2, Y2


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


def generate_neural_network_analysis():
    X_p, Y_p, _ = get_phishing_data()
    X_v, Y_v, _ = get_vocal_data()

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    hyperNN(X_train, y_train, X_test, y_test,
            title="Model Complexity Curve for NN (Phishing Data)\nHyperparameter : No. Hidden Units")
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
    h_units, learn_rate = NNGridSearchCV(X_train, y_train)
    estimator_vocal = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic',
                                    learning_rate_init=learn_rate, random_state=100)
    train_samp_vocal, NN_train_score_vocal, NN_fit_time_vocal, NN_pred_time_vocal = plot_learning_curve(estimator_vocal,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        title="Neural Net Vocal Data")
    final_classifier_evaluation(estimator_vocal, X_train, X_test, y_train, y_test)


np.random.seed(0)


def run_kmeans(X, y, title):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 30), metric='silhouette', title=title)

    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.poof()  # Draw/show/poof the data

    visualizer = KElbowVisualizer(model, k=(2, 30), metric='distortion', title=title)

    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.poof()  # Draw/show/poof the data

    visualizer = KElbowVisualizer(model, k=(2, 30), metric='calinski_harabaz', title=title)

    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.poof()  # Draw/show/poof the data


def evaluate_kmeans(km, X, y, title=""):
    start_time = timeit.default_timer()
    km.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    y_mode_vote = cluster_predictions(y, km.labels_)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote - " + title)
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(km.n_iter_))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


def kmeans_analysis():
    X_p, Y_p, _ = get_phishing_data()
    # run_kmeans(X_p, Y_p, 'Phishing Dataset')
    km = KMeans(n_clusters=4, random_state=100)
    evaluate_kmeans(km, X_p, Y_p)
    df = pd.DataFrame(km.cluster_centers_)
    df.to_csv("Phishing kMeans Cluster Centers.csv")

    X_v, Y_v, _ = get_vocal_data()
    # run_kmeans(X_v, Y_v, 'Vocal Dataset')
    km = KMeans(n_clusters=12, random_state=100)
    evaluate_kmeans(km, X_v, Y_v)
    df = pd.DataFrame(km.cluster_centers_)
    df.to_csv("Vocal Data kMeans Cluster Centers.csv")


def run_EM(X, y, title):
    # kdist =  [2,3,4,5]
    # kdist = list(range(2,51))
    kdist = list(np.arange(2, 100, 5))
    sil_scores = [];
    f1_scores = [];
    homo_scores = [];
    train_times = [];
    aic_scores = [];
    bic_scores = []

    for k in kdist:
        start_time = timeit.default_timer()
        em = EM(n_components=k, covariance_type='diag', n_init=1, warm_start=True, random_state=100).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)

        labels = em.predict(X)
        sil_scores.append(silhouette_score(X, labels))
        y_mode_vote = cluster_predictions(y, labels)
        f1_scores.append(f1_score(y, y_mode_vote))
        homo_scores.append(homogeneity_score(y, labels))
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Elbow Plot for EM: ' + title)
    plt.show()

    # plot homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, homo_scores)
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores EM: ' + title)
    plt.show()

    # plot f1 scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, f1_scores)
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores EM: ' + title)
    plt.show()

    # plot model AIC and BIC
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores, label='BIC')
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Model Complexity Score')
    plt.title('EM Model Complexity: ' + title)
    plt.legend(loc="best")
    plt.show()


def evaluate_EM(em, X, y, title=""):
    start_time = timeit.default_timer()
    em.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y, labels)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote - " + title)
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(em.n_iter_))
    print("Log-likelihood Lower Bound: {:.2f}".format(em.lower_bound_))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


def em_analysis():
    X_p, Y_p, _ = get_phishing_data()
    # run_EM(X_p, Y_p, 'Phishing Data')
    em = EM(n_components=30, covariance_type='diag', n_init=1, warm_start=True, random_state=100)
    evaluate_EM(em, X_p, Y_p)
    df = pd.DataFrame(em.means_)
    df.to_csv("Phishing EM Component Means.csv")

    X_v, Y_v, _ = get_vocal_data()
    # run_EM(X_v, Y_v, 'Vocal Data')
    em = EM(n_components=52, covariance_type='diag', n_init=1, warm_start=True, random_state=100)
    evaluate_EM(em, X_v, Y_v)
    df = pd.DataFrame(em.means_)
    df.to_csv("Vocal EM Component Means.csv")


def run_PCA(X, y, title):
    pca = PCA(random_state=5).fit(X)  # for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Explained Variance and Eigenvalues: " + title)
    fig.tight_layout()
    plt.show()


def run_ICA(X, y, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.show()


def run_RCA(X, y, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i, dim in product(range(5), dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(dims, mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims, std_recon, 'm-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: " + title)
    fig.tight_layout()
    plt.show()


def run_RFC(X, y, df_original, title="RFC"):
    rfc = RFC(n_estimators=500, random_state=5, n_jobs=-1)
    imp = rfc.fit(X, y).feature_importances_
    imp = pd.DataFrame(imp, columns=['Feature Importance'], index=df_original.columns[1::])
    imp.sort_values(by=['Feature Importance'], inplace=True, ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    top_cols = imp.index.tolist()
    # print(imp)
    return imp, top_cols


def addclusters(X, km_labels, em_labels):
    df = pd.DataFrame(X)
    df['KM Cluster'] = km_labels
    df['EM Cluster'] = em_labels
    col_1hot = ['KM Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot, axis=1)
    df = pd.concat([df_others, df_1hot], axis=1)
    new_X = np.array(df.values, dtype='int64')

    return new_X


def dimensionality_reduction_analysis():
    X_p, Y_p, df_phish = get_phishing_data()
    # run_PCA(X_p, Y_p, "Phishing Data")
    # run_ICA(X_p, Y_p, "Phishing Data")
    # run_RCA(X_p, Y_p, "Phishing Data")
    imp_phish, topcols_phish = run_RFC(X_p, Y_p, df_original=df_phish)
    pca_phish = PCA(n_components=32, random_state=5).fit_transform(X_p)
    ica_phish = ICA(n_components=32, random_state=5).fit_transform(X_p)
    rca_phish = RCA(n_components=32, random_state=5).fit_transform(X_p)
    rfc_phish = df_phish[topcols_phish]
    rfc_phish = np.array(rfc_phish.values, dtype='int64')[:, :32]
    #
    # run_kmeans(pca_phish, Y_p, 'PCA Phishing Data')
    # run_kmeans(ica_phish, Y_p, 'ICA Phishing Data')
    # run_kmeans(rca_phish, Y_p, 'RCA Phishing Data')
    # run_kmeans(rfc_phish, Y_p, 'RFC Phishing Data')
    #
    # evaluate_kmeans(KMeans(n_clusters=14, n_init=10, random_state=100, n_jobs=-1), pca_phish, Y_p, title="PCA")
    # evaluate_kmeans(KMeans(n_clusters=12, n_init=10, random_state=100, n_jobs=-1), ica_phish, Y_p, title="ICA")
    # evaluate_kmeans(KMeans(n_clusters=10, n_init=10, random_state=100, n_jobs=-1), rca_phish, Y_p, title="RCA")
    # evaluate_kmeans(KMeans(n_clusters=2, n_init=10, random_state=100, n_jobs=-1), rfc_phish, Y_p, title="RFC")
    #
    # run_EM(pca_phish, Y_p, 'PCA Phishing Data')
    # run_EM(ica_phish, Y_p, 'ICA Phishing Data')
    # run_EM(rca_phish, Y_p, 'RCA Phishing Data')
    # run_EM(rfc_phish, Y_p, 'RFC Phishing Data')
    #
    # evaluate_EM(
    #     EM(n_components=67, covariance_type='diag', n_init=1, warm_start=True, random_state=100), pca_phish, Y_p,
    #     title="PCA")
    # evaluate_EM(
    #     EM(n_components=64, covariance_type='diag', n_init=1, warm_start=True, random_state=100), ica_phish, Y_p,
    #     title="ICA")
    # evaluate_EM(
    #     EM(n_components=64, covariance_type='diag', n_init=1, warm_start=True, random_state=100), rca_phish, Y_p,
    #     title="RCA")
    # evaluate_EM(
    #     EM(n_components=32, covariance_type='diag', n_init=1, warm_start=True, random_state=100), rfc_phish, Y_p,
    #     title="RFC")
    #
    # X_v, Y_v, df_vocal = get_vocal_data()
    # run_PCA(X_v, Y_v, "Phone Me Data")
    # run_ICA(X_v, Y_v, "Phone Me Data")
    # run_RCA(X_v, Y_v, "Phone Me Data")
    # imp_vocal, topcols_vocal = run_RFC(X_v, Y_v, df_original=df_vocal)
    # pca_vocal = PCA(n_components=4, random_state=5).fit_transform(X_v)
    # ica_vocal = ICA(n_components=4, random_state=5).fit_transform(X_v)
    # rca_vocal = RCA(n_components=4, random_state=5).fit_transform(X_v)
    # rfc_vocal = df_vocal[topcols_vocal]
    # rfc_vocal = np.array(rfc_vocal.values, dtype='int64')[:, :4]
    #
    # run_kmeans(pca_vocal, Y_v, 'PCA Phone Me Data')
    # run_kmeans(ica_vocal, Y_v, 'ICA Phone Me Data')
    # run_kmeans(rca_vocal, Y_v, 'RCA Phone Me Data')
    # run_kmeans(rfc_vocal, Y_v, 'RFC Phone Me Data')
    #
    # evaluate_kmeans(KMeans(n_clusters=12, n_init=10, random_state=100, n_jobs=-1), pca_vocal, Y_v, title="PCA")
    # evaluate_kmeans(KMeans(n_clusters=10, n_init=10, random_state=100, n_jobs=-1), ica_vocal, Y_v, title="ICA")
    # evaluate_kmeans(KMeans(n_clusters=12, n_init=10, random_state=100, n_jobs=-1), rca_vocal, Y_v, title="RCA")
    # evaluate_kmeans(KMeans(n_clusters=12, n_init=10, random_state=100, n_jobs=-1), rfc_vocal, Y_v, title="RFC")
    #
    # run_EM(pca_vocal, Y_v, 'PCA Phone Me Data')
    # run_EM(ica_vocal, Y_v, 'ICA Phone Me Data')
    # run_EM(rca_vocal, Y_v, 'RCA Phone Me Data')
    # run_EM(rfc_vocal, Y_v, 'RFC Phone Me Data')
    #
    # evaluate_EM(
    #     EM(n_components=58, covariance_type='diag', n_init=1, warm_start=True,
    #        random_state=100), pca_vocal, Y_v, title="PCA")
    # evaluate_EM(
    #     EM(n_components=52, covariance_type='diag', n_init=1, warm_start=True,
    #        random_state=100), ica_vocal, Y_v, title="ICA")
    # evaluate_EM(
    #     EM(n_components=56, covariance_type='diag', n_init=1, warm_start=True,
    #        random_state=100), rca_vocal, Y_v, title="RCA")
    # evaluate_EM(
    #     EM(n_components=48, covariance_type='diag', n_init=1, warm_start=True,
    #        random_state=100), rfc_vocal, Y_v, title="RFC")

    # Comparing With NN
    # Original
    print("Original")
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_p), np.array(Y_p), test_size=0.20)
    full_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                             random_state=100)
    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train,
                                                                                                    y_train,
                                                                                                    title="Neural Net Phishing: Full")
    final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)
    # PCA
    print("PCA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(pca_phish), np.array(Y_p), test_size=0.20)
    pca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing: PCA")
    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)
    # ICA
    print("ICA")
    X_train, X_test, y_train, y_test = train_test_split(np.array(ica_phish), np.array(Y_p), test_size=0.20)
    ica_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing: ICA")
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)
    # Randomised Projection
    print("RCA")
    X_train, X_test, y_train, y_test = train_test_split(np.array(rca_phish), np.array(Y_p), test_size=0.20)
    rca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing: RCA")
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)
    # RFC
    print("RFC")
    X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_phish), np.array(Y_p), test_size=0.20)
    rfc_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing: RFC")
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)

    compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Phishing Dataset')
    compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Phishing Dataset')
    compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica,
                       NN_train_score_rca, NN_train_score_rfc, 'Phishing Dataset')

    print("Training Clustered Label")
    # Training NN on Projected data with cluster labels
    km = KMeans(n_clusters=2, n_init=10, random_state=100, n_jobs=-1).fit(X_p)
    km_labels = km.labels_
    em = EM(n_components=30, covariance_type='diag', n_init=1, warm_start=True, random_state=100).fit(X_p)
    em_labels = em.predict(X_p)

    clust_full = addclusters(X_p, km_labels, em_labels)
    clust_pca = addclusters(pca_phish, km_labels, em_labels)
    clust_ica = addclusters(ica_phish, km_labels, em_labels)
    clust_rca = addclusters(rca_phish, km_labels, em_labels)
    clust_rfc = addclusters(rfc_phish, km_labels, em_labels)
    print("Training Clustered - Original")

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_full), np.array(Y_p), test_size=0.20)
    full_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                             random_state=100)
    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train,
                                                                                                    y_train,
                                                                                                    title="Neural Net Phishing with Clusters: Full")
    final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)
    print("Training Clustered - PCA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_pca), np.array(Y_p), test_size=0.20)
    pca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing with Clusters: PCA")
    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)
    print("Training Clustered - ICA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_ica), np.array(Y_p), test_size=0.20)
    ica_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing with Clusters: ICA")
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)
    print("Training Clustered - RCA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rca), np.array(Y_p), test_size=0.20)
    rca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing with Clusters: RCA")
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)
    print("Training Clustered - RFC")

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rfc), np.array(Y_p), test_size=0.20)
    rfc_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01,
                            random_state=100)
    train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Phishing with Clusters: RFC")
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)

    compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Phishing Dataset')
    compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Phishing Dataset')
    compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica,
                       NN_train_score_rca, NN_train_score_rfc, 'Phishing Dataset')


# kmeans_analysis()
# em_analysis()
dimensionality_reduction_analysis()
