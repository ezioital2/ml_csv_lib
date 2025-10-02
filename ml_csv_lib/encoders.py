# ml_csv_lib/encoders.py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, PowerTransformer
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class Encoders:
    @staticmethod
    def label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies Label Encoding to specified columns.
        
        :param df: DataFrame.
        :param columns: List of columns to encode.
        :return: Transformed DataFrame.
        """
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies One-Hot Encoding to specified columns.
        
        :param df: DataFrame.
        :param columns: List of columns to encode.
        :return: Transformed DataFrame.
        """
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = ohe.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns), index=df.index)
        df = df.drop(columns, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        return df
    
    @staticmethod
    def ordinal_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies Ordinal Encoding to specified columns.
        
        :param df: DataFrame.
        :param columns: List of columns to encode.
        :return: Transformed DataFrame.
        """
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[columns] = oe.fit_transform(df[columns])
        return df
    
    @staticmethod
    def target_encode(df: pd.DataFrame, columns: list, target_col: str) -> pd.DataFrame:
        """
        Applies Target Encoding to specified columns using the target variable.
        
        :param df: DataFrame.
        :param columns: List of columns to encode.
        :param target_col: Target column for mean encoding.
        :return: Transformed DataFrame.
        """
        for col in columns:
            means = df.groupby(col)[target_col].mean()
            df[col] = df[col].map(means)
        return df
    
    @staticmethod
    def frequency_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies Frequency Encoding to specified columns.
        
        :param df: DataFrame.
        :param columns: List of columns to encode.
        :return: Transformed DataFrame.
        """
        for col in columns:
            freq = df[col].value_counts() / len(df)
            df[col] = df[col].map(freq)
        return df
    
    @staticmethod
    def embed_categorical(df: pd.DataFrame, columns: list, embedding_dim: int = 5) -> pd.DataFrame:
        """
        Applies embedding to categorical columns using One-Hot + PCA.
        
        :param df: DataFrame.
        :param columns: List of columns to embed.
        :param embedding_dim: Dimension of embedding.
        :return: Transformed DataFrame.
        """
        df_temp = Encoders.one_hot_encode(df[columns].copy(), columns)
        pca = PCA(n_components=embedding_dim)
        embedded = pca.fit_transform(df_temp)
        embedded_df = pd.DataFrame(embedded, columns=[f"emb_{i}" for i in range(embedding_dim)], index=df.index)
        df = df.drop(columns, axis=1)
        df = pd.concat([df, embedded_df], axis=1)
        return df