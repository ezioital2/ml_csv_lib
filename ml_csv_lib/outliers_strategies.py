from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats

class OutlierStrategy(ABC):
    @abstractmethod
    def treat_outliers(self, df: pd.DataFrame, columns: list, treatment: str = 'remove', **kwargs) -> pd.DataFrame:
        """
        Treats outliers in specified columns.
        
        :param df: DataFrame.
        :param columns: List of columns to check for outliers.
        :param treatment: 'remove', 'cap', or 'mean'.
        :param kwargs: Strategy-specific parameters.
        :return: DataFrame with outliers treated.
        """
        pass

class IQRStrategy(OutlierStrategy):
    def treat_outliers(self, df: pd.DataFrame, columns: list, treatment: str = 'remove', multiplier: float = 1.5, **kwargs) -> pd.DataFrame:
        """
        Treats outliers using Interquartile Range (IQR) method.
        
        :param df: DataFrame.
        :param columns: List of columns to apply IQR.
        :param treatment: 'remove', 'cap', or 'mean'.
        :param multiplier: Multiplier for IQR.
        :return: Treated DataFrame.
        """
        df_treated = df.copy()
        initial_rows = len(df_treated)
        
        for col in columns:
            if col not in df_treated.columns:
                print(f"⚠️ Warning: Column '{col}' not found.")
                continue
                
            Q1 = df_treated[col].quantile(0.25)
            Q3 = df_treated[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            if treatment == 'remove':
                # Remove rows with outliers
                mask = (df_treated[col] >= lower_bound) & (df_treated[col] <= upper_bound)
                df_treated = df_treated[mask]
                
            elif treatment == 'cap':
                # Cap outliers to bounds
                df_treated[col] = np.where(df_treated[col] < lower_bound, lower_bound, df_treated[col])
                df_treated[col] = np.where(df_treated[col] > upper_bound, upper_bound, df_treated[col])
                
            elif treatment == 'mean':
                # Replace outliers with mean
                col_mean = df_treated[col].mean()
                outlier_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                df_treated.loc[outlier_mask, col] = col_mean
                
            else:
                raise ValueError(f"Unknown treatment: {treatment}")
        
        final_rows = len(df_treated)
        if treatment == 'remove':
            print(f"🔹 Removed outliers using IQR. Rows before: {initial_rows}, after: {final_rows}")
        else:
            print(f"🔹 Treated outliers using IQR with '{treatment}'. Rows: {final_rows}")
            
        return df_treated

class ZScoreStrategy(OutlierStrategy):
    def treat_outliers(self, df: pd.DataFrame, columns: list, treatment: str = 'remove', threshold: float = 3.0, **kwargs) -> pd.DataFrame:
        """
        Treats outliers using Z-Score method.
        
        :param df: DataFrame.
        :param columns: List of columns to apply Z-Score.
        :param treatment: 'remove', 'cap', or 'mean'.
        :param threshold: Z-Score threshold.
        :return: Treated DataFrame.
        """
        df_treated = df.copy()
        initial_rows = len(df_treated)
        
        for col in columns:
            if col not in df_treated.columns:
                print(f"⚠️ Warning: Column '{col}' not found.")
                continue
                
            z_scores = np.abs(stats.zscore(df_treated[col].dropna()))
            
            if treatment == 'remove':
                # Remove rows with outliers
                mask = z_scores < threshold
                df_treated = df_treated[mask]
                
            elif treatment == 'cap':
                # Cap outliers to threshold in standard deviations
                col_mean = df_treated[col].mean()
                col_std = df_treated[col].std()
                lower_bound = col_mean - threshold * col_std
                upper_bound = col_mean + threshold * col_std
                
                df_treated[col] = np.where(df_treated[col] < lower_bound, lower_bound, df_treated[col])
                df_treated[col] = np.where(df_treated[col] > upper_bound, upper_bound, df_treated[col])
                
            elif treatment == 'mean':
                # Replace outliers with mean
                col_mean = df_treated[col].mean()
                outlier_mask = z_scores > threshold
                df_treated.loc[outlier_mask, col] = col_mean
                
            else:
                raise ValueError(f"Unknown treatment: {treatment}")
        
        final_rows = len(df_treated)
        if treatment == 'remove':
            print(f"🔹 Removed outliers using Z-Score. Rows before: {initial_rows}, after: {final_rows}")
        else:
            print(f"🔹 Treated outliers using Z-Score with '{treatment}'. Rows: {final_rows}")
            
        return df_treated

class PercentileStrategy(OutlierStrategy):
    def treat_outliers(self, df: pd.DataFrame, columns: list, treatment: str = 'remove', lower_percentile: float = 0.01, upper_percentile: float = 0.99, **kwargs) -> pd.DataFrame:
        """
        Treats outliers using percentile clipping.
        
        :param df: DataFrame.
        :param columns: List of columns to apply percentiles.
        :param treatment: 'remove', 'cap', or 'mean'.
        :param lower_percentile: Lower percentile threshold.
        :param upper_percentile: Upper percentile threshold.
        :return: Treated DataFrame.
        """
        df_treated = df.copy()
        initial_rows = len(df_treated)
        
        for col in columns:
            if col not in df_treated.columns:
                print(f"⚠️ Warning: Column '{col}' not found.")
                continue
                
            lower_bound = df_treated[col].quantile(lower_percentile)
            upper_bound = df_treated[col].quantile(upper_percentile)
            
            if treatment == 'remove':
                # Remove rows with outliers
                mask = (df_treated[col] >= lower_bound) & (df_treated[col] <= upper_bound)
                df_treated = df_treated[mask]
                
            elif treatment == 'cap':
                # Cap outliers to bounds
                df_treated[col] = np.where(df_treated[col] < lower_bound, lower_bound, df_treated[col])
                df_treated[col] = np.where(df_treated[col] > upper_bound, upper_bound, df_treated[col])
                
            elif treatment == 'mean':
                # Replace outliers with mean
                col_mean = df_treated[col].mean()
                outlier_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                df_treated.loc[outlier_mask, col] = col_mean
                
            else:
                raise ValueError(f"Unknown treatment: {treatment}")
        
        final_rows = len(df_treated)
        if treatment == 'remove':
            print(f"🔹 Removed outliers using Percentiles. Rows before: {initial_rows}, after: {final_rows}")
        else:
            print(f"🔹 Treated outliers using Percentiles with '{treatment}'. Rows: {final_rows}")
            
        return df_treated

# Estrategia compuesta actualizada
class CompositeOutlierStrategy(OutlierStrategy):
    def __init__(self, strategies: list):
        self.strategies = strategies
    
    def treat_outliers(self, df: pd.DataFrame, columns: list, treatment: str = 'remove', **kwargs) -> pd.DataFrame:
        df_treated = df.copy()
        for strategy in self.strategies:
            df_treated = strategy.treat_outliers(df_treated, columns, treatment, **kwargs)
        return df_treated