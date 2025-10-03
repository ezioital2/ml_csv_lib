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
    
    # ⭐⭐ NEW: Method to drop specific columns
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.
        
        :param df: DataFrame.
        :param columns_to_drop: List of column names to drop.
        :return: DataFrame with columns dropped.
        """
        missing_cols = [col for col in columns_to_drop if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Warning: Columns not found and skipped: {missing_cols}")
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        return df.drop(existing_cols, axis=1)
    
    # ⭐⭐ NEW: Method to drop specific rows by indices
    @staticmethod
    def drop_rows(df: pd.DataFrame, row_indices: list) -> pd.DataFrame:
        """
        Drops specified rows from the DataFrame by their indices.
        
        :param df: DataFrame.
        :param row_indices: List of row indices to drop.
        :return: DataFrame with rows dropped.
        """
        invalid_indices = [idx for idx in row_indices if idx not in df.index]
        if invalid_indices:
            print(f"⚠️ Warning: Invalid row indices skipped: {invalid_indices}")
        valid_indices = [idx for idx in row_indices if idx in df.index]
        return df.drop(valid_indices, axis=0)
    
    # ⭐⭐ NEW: Method to drop rows based on a condition (more flexible)
    @staticmethod
    def drop_rows_by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Drops rows from the DataFrame based on a pandas query condition.
        
        Example: condition = 'age > 100' to drop rows where age > 100.
        
        :param df: DataFrame.
        :param condition: String representing the condition (uses df.query).
        :return: DataFrame with rows dropped.
        """
        try:
            return df.query(f"not ({condition})")
        except Exception as e:
            raise ValueError(f"Invalid condition: {condition}. Error: {str(e)}")