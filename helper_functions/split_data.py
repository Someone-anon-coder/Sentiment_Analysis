import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(filename: str) -> None:
    """Split the dataset into training and test sets

    Args:
        filename (str): Name of the dataset to be split

    Returns:
        None
    """
    
    df = pd.read_csv(filename)
    df.dropna(inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv('New_Data/train_split.csv', index=False)
    test_df.to_csv('New_Data/test_split.csv', index=False)

    print("Training and test datasets have been saved as 'train_split.csv' and 'test_split.csv'.")

if __name__ == "__main__":
    filename = "Data/new_train_preprocessed.csv"
    
    split_data(filename)