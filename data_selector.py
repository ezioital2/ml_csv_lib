# ml_csv_lib/data_selector.py
import pandas as pd

class DataSelector:
    @staticmethod
    def select_x_y(df: pd.DataFrame, target_col: str):
        """
        Selects X (features) and y (target) based on target column name.
        
        :param df: DataFrame.
        :param target_col: Name of the target variable.
        :return: X (DataFrame), y (Series).
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        return X, y