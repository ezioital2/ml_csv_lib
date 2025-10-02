from abc import ABC, abstractmethod
import pandas as pd
from .treatments import SupervisedTreatmentType, UnsupervisedTreatmentType, RegressionTreatmentType, apply_supervised_treatment, apply_unsupervised_treatment, apply_regression_treatment

class TreatmentStrategy(ABC):
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, treatment_type, cat_cols: list, cont_cols: list, **kwargs):
        pass

class SupervisedStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, treatment_type: SupervisedTreatmentType, cat_cols: list, cont_cols: list, scaled: bool = True, **kwargs) -> tuple:
        """
        Prepares data for supervised learning.
        
        :param df: DataFrame.
        :param treatment_type: SupervisedTreatmentType enum.
        :param cat_cols: Categorical columns.
        :param cont_cols: Continuous columns.
        :param scaled: Whether to scale continuous variables.
        :param kwargs: Must include 'target_col'.
        :return: X, y after treatment.
        """
        if 'target_col' not in kwargs:
            raise ValueError("Supervised strategy requires 'target_col' in kwargs.")
        target_col = kwargs['target_col']
        
        # Remove target from cat/cont if present
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        if target_col in cont_cols:
            cont_cols.remove(target_col)
        
        transformed_df = apply_supervised_treatment(df, treatment_type, cat_cols, cont_cols, target_col, scaled)
        X, y = transformed_df.drop(target_col, axis=1), transformed_df[target_col]
        return X, y

class UnsupervisedStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, treatment_type: UnsupervisedTreatmentType, cat_cols: list, cont_cols: list, scaled: bool = True, **kwargs) -> pd.DataFrame:
        """
        Prepares data for unsupervised learning (e.g., clustering).
        Handles mixed types for clustering.
        
        :param df: DataFrame.
        :param treatment_type: UnsupervisedTreatmentType enum.
        :param cat_cols: Categorical columns.
        :param cont_cols: Continuous columns.
        :param scaled: Whether to scale continuous variables.
        :return: Transformed DataFrame ready for clustering.
        """
        return apply_unsupervised_treatment(df, treatment_type, cat_cols, cont_cols, scaled)

class RegressionStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, treatment_type: RegressionTreatmentType, cat_cols: list, cont_cols: list, target_col: str, additional_transform: str = 'none', scaled: bool = True) -> tuple:
        """
        Prepares data for regression, with optional additional transformation (yeo-johnson, box-cox, none).
        
        :param df: DataFrame.
        :param treatment_type: RegressionTreatmentType enum.
        :param cat_cols: Categorical columns.
        :param cont_cols: Continuous columns.
        :param target_col: Name of the target variable.
        :param additional_transform: 'yeo-johnson', 'box-cox', or 'none'.
        :param scaled: Whether to scale continuous variables.
        :return: X, y after treatment.
        """
        # Remove target from cat/cont if present
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        if target_col in cont_cols:
            cont_cols.remove(target_col)
        
        transformed_df = apply_regression_treatment(df, treatment_type, cat_cols, cont_cols, target_col, additional_transform, scaled)
        X, y = transformed_df.drop(target_col, axis=1), transformed_df[target_col]
        return X, y
