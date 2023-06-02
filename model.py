import numpy as np
from tqdm import tqdm
from typing import Tuple


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X) -> None:
        # fit the PCA model

        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # compute the covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # compute the eigenvalues and eigenvectors of A
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort the eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        top_k_eigenvectors = eigenvectors[:,:self.n_components]
        self.components = top_k_eigenvectors

    
    def transform(self, X) -> np.ndarray:
        # transform the data

        # center the data
        X_centered = X - self.mean

        # project the data onto the new basis
        X_transformed = np.dot(X_centered, self.components)
        
        return X_transformed

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # Define the loss function 
        def hinge_loss(xi, yi):
            return max(0, 1 - yi * (np.dot(self.w, xi) + self.b))

       # Define the hinge loss gradiend function for updation of parameters
        def hinge_loss_grad(xi, yi):
            if hinge_loss(xi, yi) == 0:
                return self.w, 0
            else:
                return (self.w + (C * -1 *  yi * xi)), (C * yi* -1)

        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):

             # sample a random training example
            j = np.random.randint(X.shape[0])
            X_j, y_j = X[j], y[j]
            
            # compute the gradient
            if y_j * (np.dot(self.w, X_j) + self.b) < 1:

                # Compute the gradient of the loss
                dw, db = hinge_loss_grad(X_j, y_j)
                
                # Update the weights and bias
                self.w -= learning_rate * dw
                self.b -= learning_rate * db
                   
    
    def predict(self, X) -> np.ndarray:
        return (np.dot(X,self.w)+self.b)
    
    def predict_class(self,X) -> np.ndarray:
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict_class(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.precision= None
        self.recall = None 
        self.f1 = None
        self.accuracy = None
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())

    def preprocess_data(self, X, y, class_label) -> Tuple[np.ndarray, np.ndarray]:
        # preprocess the data to make it suitable for the 1-vs-rest SVM model
        # set the positive class label to 1 and the negative class labels to -1
        X_new = X.copy()
        y_new = np.where(y == class_label, 1, -1)
        return X_new, y_new
    
    def fit(self, X, y, **kwargs) -> None:
        C, learning_rate, num_iters = kwargs.values()

        # train the 10 SVM models using the preprocessed data for each class
        for i in range(self.num_classes):
            
            # preprocess the data for the i-th class
            X_new, y_new = self.preprocess_data(X, y, i)

            # train the SVM model for the i-th class
            self.models[i].fit(X_new, y_new, learning_rate, num_iters, C)

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            scores[:, i] = self.models[i].predict(X)
        return np.argmax(scores, axis=1)

    def accuracy_score(self, X, y) -> float:
        self.accuracy = np.mean(self.predict(X) == y) 
        return self.accuracy
    
    def precision_score(self, X, y) -> float:
        # calculate precision score for each class separately
        precisions = []
        for i in range(self.num_classes):
            y_binary = np.where(y == i, 1, -1)
            predictions = self.models[i].predict_class(X)
            true_positives = np.sum((predictions == 1) & (y_binary == 1))
            total_positives = np.sum(predictions == 1)
            if total_positives == 0:
                precisions.append(0.0)
            else:
                precisions.append(true_positives / total_positives)
        
        self.precision = np.mean(precisions)
        # return the average precision score across all classes
        return self.precision
    
    def recall_score(self, X, y) -> float:
        # calculate the recall score for each class
        recalls = []
        for i in range(self.num_classes):
            y_binary = np.where(y == i, 1, -1)
            predictions = self.models[i].predict_class(X)
            true_positives = np.sum((predictions == 1) & (y_binary == 1))
            false_negatives = np.sum((predictions == -1) & (y_binary == 1))
            
            if (true_positives + false_negatives) != 0:
                recalls.append(true_positives / (true_positives + false_negatives))
            else:
                recalls.append(0.0)
        
        # calculate the average recall score
        self.recall = np.mean(recalls)
        return self.recall
    
    def f1_score(self, X, y) -> float:
        # compute precision and recall scores
        precision = self.precision if self.precision!=None else self.precision_score(X, y)
        recall = self.recall if self.recall!=None else self.recall_score(X, y)
        
        # compute the F1 score
        if precision + recall == 0:
            self.f1 = 0
        else:
            self.f1 = 2 * (precision * recall) / (precision + recall)
        
        return self.f1
