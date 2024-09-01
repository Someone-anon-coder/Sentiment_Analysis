from pandas import DataFrame
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from train_model import load_data

def define_parameter_grid() -> dict:
    """Define the parameter grid for hyperparameter tuning

    Args:
        None

    Returns:
        dict: Parameter grid for hyperparameter tuning
    """
    
    param_grid = {
        'alpha': [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
        'fit_prior': [True, False],
    }
    return param_grid

def tune_hyperparameters(X_train: DataFrame, y_train: DataFrame, model_file: str) -> None:
    """Tuning Model Hyperparameters

    Args:
        X_train (DataFrame): Training data input features
        y_train (DataFrame): Training data output labels
        model_file (str): Model file to be saved

    Returns:
        None
    """
    
    param_grid = define_parameter_grid()
    model = MultinomialNB()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    with open(f'Pkl_Files/{model_file}', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    print(f"Best model saved as '{model_file}'.")

def main(train_pkl_file: str, test_pkl_file: str, train_csv_file: str, test_csv_file: str, model_file: str) -> None:
    """Tuning Model Hyperparameters

    Args:
        train_pkl_file (str): Training pkl file
        test_pkl_file (str): Test pkl file
        train_csv_file (str): Training csv file
        test_csv_file (str): Test csv file
        model_file (str): Model file to be saved

    Returns:
        None
    """
    
    X_train, _, y_train, _ = load_data(train_pkl_file, test_pkl_file, train_csv_file, test_csv_file)
    tune_hyperparameters(X_train, y_train, model_file)

if __name__ == "__main__":
    train_pkl_file = 'Pkl_Files/X_train.pkl'
    test_pkl_file = 'Pkl_Files/X_test.pkl'
    train_csv_file = 'New_Data/train_split.csv'
    test_csv_file = 'New_Data/test_split.csv'
    
    model_file = 'best_naive_bayes_model.pkl'
    
    main(train_pkl_file, test_pkl_file, train_csv_file, test_csv_file, model_file)
