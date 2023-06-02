from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple


def get_hyperparameters() -> Tuple[float, int, float]:

    # learning rate (learning_rate)
    # number of iterations (num_iters)
    # # regularization parameter (C)

    # learning_rate,num_iters,C = (0.0001,500000,10.0) # max accuracy is 91.31% at k = 500
    # learning_rate,num_iters,C = (0.001,50000,10.0) # max accuracy is 82.11% at k = 100
    # learning_rate,num_iters,C = (0.00001,60000,10.0) # max accuracy is 91.04% at k = 500
    learning_rate,num_iters,C = (0.00001,500000,100.0) # max accuracy is 91.31% at k = 500
    return learning_rate, num_iters, C


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    metrics = []
    for k in [2,5, 10, 20, 50, 100, 200,500]:
        # reduce the dimensionality of the data
        pca = PCA(n_components=k)
        X_train_emb = pca.fit_transform(X_train)
        X_test_emb = pca.transform(X_test)

        # create a model
        svm = MultiClassSVM(num_classes=10)

        # fit the model
        svm.fit(
            X_train_emb, y_train, C=C,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

        # evaluate the model
        accuracy = svm.accuracy_score(X_test_emb, y_test)
        precision = svm.precision_score(X_test_emb, y_test)
        recall = svm.recall_score(X_test_emb, y_test)
        f1_score = svm.f1_score(X_test_emb, y_test)
        
        metrics.append((k, accuracy, precision, recall, f1_score,svm))
        
        parameters = (learning_rate,num_iters,C)

        print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')

    # plot and save the results
    plot_metrics(metrics,parameters)


if __name__ == '__main__':
    main()
