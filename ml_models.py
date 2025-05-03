import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

def train_model(X_train, y_train, model_type='Linear Regression'):
    """
    Train machine learning model based on selected type.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    model_type : str, default='Linear Regression'
        Type of model to train (Linear Regression, Logistic Regression, K-Means Clustering)
    
    Returns:
    --------
    tuple (model, feature_importance)
        - model: The trained model
        - feature_importance: Feature importance scores (if available)
    """
    if model_type == 'Linear Regression':
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Get feature importance (coefficients)
        feature_importance = pd.Series(model.coef_, index=X_train.columns)
        
    elif model_type == 'Logistic Regression':
        # For Logistic Regression, we need to convert the target to binary
        # Using the median as threshold
        threshold = np.median(y_train)
        y_train_binary = (y_train >= threshold).astype(int)
        
        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train_binary)
        
        # Get feature importance (coefficients)
        feature_importance = pd.Series(model.coef_[0], index=X_train.columns)
        
    elif model_type == 'K-Means Clustering':
        # For K-Means, we combine features and target for unsupervised learning
        X_kmeans = X_train.copy()
        
        # Add target as a feature
        X_kmeans['target'] = y_train.values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_kmeans)
        
        # Determine optimal number of clusters (using simplified approach)
        inertia = []
        K_range = range(2, min(10, len(X_scaled) // 5 + 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        k_optimal = 3  # Default
        if len(K_range) > 2:
            inertia_diff = np.diff(inertia)
            inertia_diff2 = np.diff(inertia_diff)
            k_optimal = K_range[np.argmax(inertia_diff2) + 1]
        
        # Train K-Means with optimal clusters
        model = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        model.fit(X_scaled)
        
        # For K-Means, feature importance is not directly available
        # We'll use the cluster centers as a proxy for feature importance
        feature_importance = pd.Series(
            np.std(model.cluster_centers_, axis=0),
            index=X_kmeans.columns
        )
        
        # Store the scaler for future use
        model.scaler = scaler
        model.feature_names = X_kmeans.columns
    
    return model, feature_importance

def evaluate_model(model, X_test, y_test, model_type='Linear Regression'):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    model_type : str, default='Linear Regression'
        Type of model to evaluate
    
    Returns:
    --------
    tuple (evaluation_metrics, predictions)
        - evaluation_metrics: Dictionary of evaluation metrics
        - predictions: Model predictions on test data
    """
    evaluation_metrics = {}
    
    if model_type == 'Linear Regression':
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        evaluation_metrics['regression_metrics'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
    elif model_type == 'Logistic Regression':
        # For Logistic Regression, convert true values to binary using the same threshold logic
        threshold = np.median(y_test)
        y_test_binary = (y_test >= threshold).astype(int)
        
        # Make probability predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Make class predictions
        y_pred = model.predict(X_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, zero_division=0)
        recall = recall_score(y_test_binary, y_pred, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test_binary, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        evaluation_metrics['classification_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score
        }
        
        # Set predictions to probability scores for visualization
        predictions = y_pred_proba
        
    elif model_type == 'K-Means Clustering':
        # For K-Means, we need to combine features and target
        X_kmeans = X_test.copy()
        X_kmeans['target'] = y_test.values
        
        # Scale the data using the same scaler
        X_scaled = model.scaler.transform(X_kmeans)
        
        # Get cluster assignments
        cluster_labels = model.predict(X_scaled)
        
        # Calculate clustering metrics
        try:
            silhouette = silhouette_score(X_scaled, cluster_labels)
        except:
            silhouette = 0  # In case of single-element clusters
        
        inertia = model.inertia_
        
        # Create PCA projection for visualization (2 components)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        evaluation_metrics['clustering_metrics'] = {
            'silhouette': silhouette,
            'inertia': inertia,
            'n_clusters': model.n_clusters,
            'cluster_labels': cluster_labels,
            'pca_result': pca_result
        }
        
        # Set predictions to cluster labels for visualization
        predictions = cluster_labels
    
    return evaluation_metrics, predictions

def predict_with_model(model, data, model_type='Linear Regression'):
    """
    Make predictions with a trained model on new data.
    
    Parameters:
    -----------
    model : object
        Trained model
    data : pandas.DataFrame
        New data to predict on
    model_type : str, default='Linear Regression'
        Type of model
    
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    if model_type == 'K-Means Clustering':
        # For K-Means, scale the data
        X_scaled = model.scaler.transform(data)
        return model.predict(X_scaled)
    else:
        # For regression/classification models
        return model.predict(data)
