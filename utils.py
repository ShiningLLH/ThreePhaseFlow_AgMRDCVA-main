import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def center_data(X):
    """
    Standardize by column
    :param X，shape = (n_samples, n_features)
    """
    X_means = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_means) / X_std

    return X_norm, X_means, X_std

def normalize_data(train_data, test_data):
    """
    :param train_data，shape = (n_samples, n_features)
    :param test_data，shape = (n_samples, n_features)
    """
    train_data, X_means, X_std = center_data(train_data)
    # Standardize test data using the mean and std of training data
    test_data = (test_data - X_means) / X_std

    return train_data, test_data

def calculate_laplacian_matrix(X, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X)
    adj_matrix = knn.kneighbors_graph(X).toarray()
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix

    return torch.tensor(laplacian_matrix)