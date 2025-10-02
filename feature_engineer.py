# ml_csv_lib/feature_engineer.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureEngineer:
    @staticmethod
    def scale_continuous(df: pd.DataFrame, columns: list = None, method: str = 'standard') -> pd.DataFrame:
        """
        Scales continuous variables.
        
        :param df: DataFrame.
        :param columns: List of continuous columns. If None, scales all numeric columns.
        :param method: 'standard' or 'minmax'.
        :return: Transformed DataFrame.
        """
        if columns is None:
            columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method.")
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    @staticmethod
    def create_interactions(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        """
        Creates interaction feature between two columns.
        
        :param df: DataFrame.
        :param col1: First column.
        :param col2: Second column.
        :return: DataFrame with new interaction column.
        """
        new_col = f"{col1}_x_{col2}"
        df[new_col] = df[col1] * df[col2]
        return df
    
    @staticmethod
    def bin_continuous(df: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
        """
        Bins a continuous variable into categories.
        
        :param df: DataFrame.
        :param column: Column to bin.
        :param bins: Number of bins.
        :return: DataFrame with binned column.
        """
        new_col = f"{column}_binned"
        df[new_col] = pd.cut(df[column], bins=bins)
        return df