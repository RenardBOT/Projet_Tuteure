from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def gridsearch_LR(X_train,X_test,y_train,y_test):
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['saga'],
        'max_iter': [100, 500, 1000],
        'tol': [0.0001, 0.0005, 0.001]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)
    print("Entraînement avec gridsearch...")
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres: ", grid.best_params_)
    return grid