import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import gridsearch

from tqdm import tqdm

# Récupération du fichier de configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# ARGUMENT PARSING. 
def parse_args():
    parser = argparse.ArgumentParser(description='Utilise un modèle d\'apprentissage automatique pour classifier les bots et les humains en fonction de leurs noms')
    show_or_train = parser.add_mutually_exclusive_group(required=True)
    show_or_train.add_argument('--show', action='store_true', help='Affiche les ensembles de données disponibles.')
    show_or_train.add_argument('--dataset',type=str, help='Nom de l\'ensemble de données à utiliser pour entraîner le modèle sans l\'extension.')
    parser.add_argument('--grid', action='store_true', help='Lance la gridsearch pour trouver les meilleurs paramètres.')
    parser.add_argument('--classifier', type=str, help='Classifieur à utiliser parmi : RF (Random Forest), LR (Logistic Regression), NB (Naive Bayes), SVM (Support Vector Machine), DT (Decision Tree)', choices=['RF', 'LR', 'NB', 'SVM', 'DT'])
    args = parser.parse_args()

    if args.show:
        print_datasets()
        sys.exit(0)
    else:
        if not args.classifier:
            print("Erreur : Vous devez spécifier un classifieur à utiliser avec --classifier")
            sys.exit(1)
        dataset_path = os.path.join(formatted_datasets_path, args.dataset + ".csv")
        if not os.path.exists(dataset_path):
            print("Error: The dataset", args.dataset, "does not exist.")
            print_datasets()
            sys.exit(1)
        return args

def print_datasets():
    print("Available datasets:")
    for dataset in Path(formatted_datasets_path).iterdir():
        print("  -", dataset.name.split(".")[0])

def load_dataset(dataset_path):
    csv_path = os.path.join(formatted_datasets_path, dataset_path + ".csv")
    df = pd.read_csv(csv_path)
    return df

def split(dataframe):

    X = dataframe.drop(["label", "user_id"], axis=1)
    y = dataframe["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    return X_train, X_test, y_train, y_test

def train_RF(X_train, y_train, X_test, y_test, grid=False):
    if grid:
        grid_search = gridsearch.gridsearch_RF(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print_results(X_train, y_train, X_test, y_test, y_pred, model)

def train_LR(X_train, y_train, X_test, y_test, grid=False):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if grid:
        grid_search = gridsearch.gridsearch_LR(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
    else:
        model = LogisticRegression(C=0.1, max_iter=100, solver='saga', tol=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print_results(X_train, y_train, X_test, y_test, y_pred, model)

def train_DT(X_train, y_train, X_test, y_test, grid=False):
    if grid:
        grid_search = gridsearch.gridsearch_DT(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
    else:
        model = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print_results(X_train, y_train, X_test, y_test, y_pred, model)


def train_NB(X_train, y_train, X_test, y_test, grid=False):
    if grid:
        grid_search = gridsearch.gridsearch_NB(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print_results(X_train, y_train, X_test, y_test, y_pred, model)

def train_SVM(X_train, y_train, X_test, y_test, grid=False):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if grid:
        grid_search = gridsearch.gridsearch_SVM(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
    else:
        model = SVC(C=1, kernel='rbf', gamma='scale')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print_results(X_train, y_train, X_test, y_test, y_pred, model)

def print_results(X_train, y_train, X_test, y_test, y_pred, model):
    print("-- MODELE --- ")
    print(" ")
    print(model)
    print(" ")
    print("-- TRAIN & TEST --- ")
    print(" ")
    print("Train set : ", X_train.shape)
    print("Test set : ", X_test.shape)
    print(" ")
    print("-- CONFUSION MATRIX --- ")
    print(" ")
    print(pd.crosstab(y_test, y_pred, rownames=['Reel'], colnames=['Prediction'], margins=True))
    print(" ")
    print("-------------------")
    print(" ")
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1-score : ", f1_score(y_test, y_pred, average='weighted'))
    print("Recall : ", recall_score(y_test, y_pred, average='weighted'))
    print("Precision : ", precision_score(y_test, y_pred, average='weighted',zero_division=0))
    
if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    formatted_datasets_path = os.path.join(formatted_datasets_path, "mixed_users")

    args = parse_args()

    dataframe = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = split(dataframe)
    if args.classifier == "RF":
        train_RF(X_train, y_train, X_test, y_test, grid=args.grid)
    if args.classifier == "LR":
        train_LR(X_train, y_train, X_test, y_test, grid=args.grid)
    if args.classifier == "NB":
        train_NB(X_train, y_train, X_test, y_test, grid=args.grid)
    if args.classifier == "SVM":
        train_SVM(X_train, y_train, X_test, y_test, grid=args.grid)
    if args.classifier == "DT":
        train_DT(X_train, y_train, X_test, y_test, grid=args.grid)