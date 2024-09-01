import pandas as pd

def clean_dataset(filename: str, new_filename: str) -> None:
    """Cleans the dataset to proper format

    Args:
        filename (str): Name of the dataset to be cleaned 
        new_filename (str): Name of the new dataset to be created
    
    Returns:
        None
    """
    
    df = pd.read_csv(filename)
    
    df = df[['text', 'sentiment']]
    df = df[df['sentiment'] != 2]
    df['sentiment'] = df['sentiment'].replace(4, 1)

    df.to_csv(new_filename, index=False)

if __name__ == "__main__":
    filename = "Data/train.csv"
    new_filename = "Data/new_train.csv"
    
    clean_dataset(filename, new_filename)
    
    