import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# import naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def gridsearch_LR(X_train,y_train):
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['saga'],
        'max_iter': [100, 500, 1000],
        'tol': [0.0001, 0.0005, 0.001]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid

def gridsearch_NB(X_train,y_train):
    # Grid search
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    grid = GridSearchCV(GaussianNB(), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid

def gridsearch_SVM(X_train,y_train):
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid

def gridsearch_DT(X_train,y_train):
    # Grid search
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Critère pour mesurer la qualité de la division
        'max_depth': [None, 10, 20, 30],    # Profondeur maximale de l'arbre
        'min_samples_split': [2, 5, 10],    # Nombre minimum d'échantillons requis pour diviser un nœud interne
        'min_samples_leaf': [1, 2, 4]       # Nombre minimum d'échantillons requis pour être une feuille
    }

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid

def gridsearch_RF(X_train,y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],        # Nombre d'arbres dans la forêt
        'criterion': ['gini', 'entropy'],      # Critère pour mesurer la qualité de la division
        'max_depth': [None, 10, 20, 30],       # Profondeur maximale de chaque arbre
        'min_samples_split': [2, 5, 10],       # Nombre minimum d'échantillons requis pour diviser un nœud interne
        'min_samples_leaf': [1, 2, 4]          # Nombre minimum d'échantillons requis pour être une feuille
    }

    grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid