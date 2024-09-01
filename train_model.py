import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(train_pkl_file: str, test_pkl_file: str, train_csv_file: str, test_csv_file: str) -> tuple:
    """Load the data

    Args:
        train_pkl_file (str, optional): Path to the training data pickle file.
        test_pkl_file (str, optional): Path to the test data pickle file.
        train_csv_file (str, optional): Path to the training data CSV file.
        test_csv_file (str, optional): Path to the test data CSV file.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    
    with open(train_pkl_file, 'rb') as f:
        X_train = pickle.load(f)
    
    with open(test_pkl_file, 'rb') as f:
        X_test = pickle.load(f)
    
    train_df = pd.read_csv(train_csv_file)
    test_df = pd.read_csv(test_csv_file)
    
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """Train and evaluate the model

    Args:
        X_train (pd.DataFrame): Training data input features
        X_test (pd.DataFrame): Test data input features
        y_train (pd.DataFrame): Training data output labels
        y_test (pd.DataFrame): Test data output labels

    Returns:
        None
    """
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    with open('Pkl_Files/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as 'naive_bayes_model.pkl'.")

if __name__ == "__main__":
    train_pkl_file = 'Pkl_Files/X_train.pkl'
    test_pkl_file = 'Pkl_Files/X_test.pkl'
    train_csv_file = 'New_Data/train_split.csv'
    test_csv_file = 'New_Data/test_split.csv'

    X_train, X_test, y_train, y_test = load_data(train_pkl_file, test_pkl_file, train_csv_file, test_csv_file)
    train_and_evaluate_model(X_train, X_test, y_train, y_test)