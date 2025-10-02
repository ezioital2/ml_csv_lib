# ml_csv_lib/data_loader.py
import pandas as pd

class DataLoader:
    @staticmethod
    def load_csv(file_path: str, separator: str = ',') -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.
        
        :param file_path: Path to the CSV file.
        :param separator: Separator used in the CSV (default ',').
        :return: DataFrame.
        """
        return pd.read_csv(file_path, sep=separator)
    
    @staticmethod
    def print_variable_types(df: pd.DataFrame) -> None:
        """
        Prints the lists of categorical and continuous variables based on dtypes.
        Categorical: object, category, bool.
        Continuous: int, float.
        
        User can copy and manually adjust these lists.
        """
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        cont_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
        print("Categorical variables:", cat_cols)
        print("Continuous variables:", cont_cols)