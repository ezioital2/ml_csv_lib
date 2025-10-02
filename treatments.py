from enum import Enum
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from .encoders import Encoders
from .feature_engineer import FeatureEngineer

class SupervisedTreatmentType(Enum):
    ALL_TO_NUMERIC = 1  # Label encode all categoricals
    ONEHOT = 2          # One-hot categoricals
    ORDINAL = 3         # Ordinal encode categoricals
    ONEHOT_INTERACT = 5 # One-hot categoricals + interactions
    TARGET_ENCODE = 6   # Target encode categoricals

class UnsupervisedTreatmentType(Enum):
    ALL_TO_NUMERIC = 1  # Label encode all categoricals
    ONEHOT = 2          # One-hot categoricals
    ORDINAL = 3         # Ordinal encode categoricals
    LABEL_BIN = 4       # Label encode categoricals + bin continuous
    ONEHOT_INTERACT = 5 # One-hot categoricals + interactions
    FREQUENCY_ENCODE = 7  # Frequency encode categoricals
    EMBED = 8           # Embed categoricals (OneHot + PCA)

class RegressionTreatmentType(Enum):
    NONE = 0            # ⭐⭐ NUEVO: No encoding, solo transformación opcional + scaling
    ALL_TO_NUMERIC = 1  # Label encode all categoricals
    ONEHOT = 2          # One-hot categoricals
    ORDINAL = 3         # Ordinal encode categoricals
    ONEHOT_INTERACT = 5 # One-hot categoricals + interactions
    TARGET_ENCODE = 6   # Target encode categoricals

def apply_supervised_treatment(df: pd.DataFrame, treatment_type: SupervisedTreatmentType, cat_cols: list, cont_cols: list, target_col: str, scaled: bool = True) -> pd.DataFrame:
    """
    Applies the selected treatment for supervised learning.
    
    :param df: DataFrame.
    :param treatment_type: SupervisedTreatmentType enum.
    :param cat_cols: Categorical columns.
    :param cont_cols: Continuous columns.
    :param target_col: Target column for encodings like target_encode.
    :param scaled: Whether to scale continuous variables.
    :return: Transformed DataFrame.
    """
    df = df.copy()
    original_cont_cols = cont_cols.copy()
    
    if treatment_type == SupervisedTreatmentType.ALL_TO_NUMERIC:
        df = Encoders.label_encode(df, cat_cols)
    
    elif treatment_type == SupervisedTreatmentType.ONEHOT:
        df = Encoders.one_hot_encode(df, cat_cols)
    
    elif treatment_type == SupervisedTreatmentType.ORDINAL:
        df = Encoders.ordinal_encode(df, cat_cols)
    
    elif treatment_type == SupervisedTreatmentType.ONEHOT_INTERACT:
        df = Encoders.one_hot_encode(df, cat_cols)
        if len(original_cont_cols) >= 2:
            df = FeatureEngineer.create_interactions(df, original_cont_cols[0], original_cont_cols[1])
    
    elif treatment_type == SupervisedTreatmentType.TARGET_ENCODE:
        df = Encoders.target_encode(df, cat_cols, target_col)
    
    else:
        raise ValueError("Invalid supervised treatment type.")
    
    # Aplicar scaling solo si scaled=True
    if scaled and original_cont_cols:
        df = FeatureEngineer.scale_continuous(df, columns=None)
    
    return df

def apply_unsupervised_treatment(df: pd.DataFrame, treatment_type: UnsupervisedTreatmentType, cat_cols: list, cont_cols: list, scaled: bool = True) -> pd.DataFrame:
    """
    Applies the selected treatment for unsupervised learning.
    
    :param df: DataFrame.
    :param treatment_type: UnsupervisedTreatmentType enum.
    :param cat_cols: Categorical columns.
    :param cont_cols: Continuous columns.
    :param scaled: Whether to scale continuous variables.
    :return: Transformed DataFrame.
    """
    df = df.copy()
    original_cont_cols = cont_cols.copy()
    
    if treatment_type == UnsupervisedTreatmentType.ALL_TO_NUMERIC:
        df = Encoders.label_encode(df, cat_cols)
    
    elif treatment_type == UnsupervisedTreatmentType.ONEHOT:
        df = Encoders.one_hot_encode(df, cat_cols)
    
    elif treatment_type == UnsupervisedTreatmentType.ORDINAL:
        df = Encoders.ordinal_encode(df, cat_cols)
    
    elif treatment_type == UnsupervisedTreatmentType.LABEL_BIN:
        df = Encoders.label_encode(df, cat_cols)
        binned_cols = []
        for col in original_cont_cols:
            df = FeatureEngineer.bin_continuous(df, col)
            binned_cols.append(f"{col}_binned")
        df = Encoders.label_encode(df, binned_cols)
    
    elif treatment_type == UnsupervisedTreatmentType.ONEHOT_INTERACT:
        df = Encoders.one_hot_encode(df, cat_cols)
        if len(original_cont_cols) >= 2:
            df = FeatureEngineer.create_interactions(df, original_cont_cols[0], original_cont_cols[1])
    
    elif treatment_type == UnsupervisedTreatmentType.FREQUENCY_ENCODE:
        df = Encoders.frequency_encode(df, cat_cols)
    
    elif treatment_type == UnsupervisedTreatmentType.EMBED:
        df = Encoders.embed_categorical(df, cat_cols)
    
    else:
        raise ValueError("Invalid unsupervised treatment type.")
    
    # Aplicar scaling solo si scaled=True
    if scaled and original_cont_cols:
        df = FeatureEngineer.scale_continuous(df, columns=None)
    
    return df

def apply_regression_treatment(df: pd.DataFrame, treatment_type: RegressionTreatmentType, cat_cols: list, cont_cols: list, target_col: str, additional_transform: str = 'none', scaled: bool = True) -> pd.DataFrame:
    """
    Applies the selected treatment for regression, with optional power transform.
    
    :param df: DataFrame.
    :param treatment_type: RegressionTreatmentType enum.
    :param cat_cols: Categorical columns.
    :param cont_cols: Continuous columns.
    :param target_col: Target column for encodings like target_encode.
    :param additional_transform: 'yeo-johnson', 'box-cox', or 'none'.
    :param scaled: Whether to scale continuous variables.
    :return: Transformed DataFrame.
    """
    df = df.copy()
    original_cont_cols = cont_cols.copy()
    
    # Aplicar transformación de potencia si se especifica
    if additional_transform != 'none':
        pt = PowerTransformer(method=additional_transform, standardize=False)
        df[original_cont_cols] = pt.fit_transform(df[original_cont_cols])
    
    # ⭐⭐ NUEVO: Tratamiento NONE - no aplica encoding a categóricas
    if treatment_type == RegressionTreatmentType.NONE:
        # No aplica ningún encoding a las variables categóricas
        # Las variables categóricas se mantienen como están (el usuario debe manejarlas por separado)
        # Solo aplica transformación de potencia (si se especificó) y scaling (si scaled=True)
        pass
    
    elif treatment_type == RegressionTreatmentType.ALL_TO_NUMERIC:
        df = Encoders.label_encode(df, cat_cols)
    
    elif treatment_type == RegressionTreatmentType.ONEHOT:
        df = Encoders.one_hot_encode(df, cat_cols)
    
    elif treatment_type == RegressionTreatmentType.ORDINAL:
        df = Encoders.ordinal_encode(df, cat_cols)
    
    elif treatment_type == RegressionTreatmentType.ONEHOT_INTERACT:
        df = Encoders.one_hot_encode(df, cat_cols)
        if len(original_cont_cols) >= 2:
            df = FeatureEngineer.create_interactions(df, original_cont_cols[0], original_cont_cols[1])
    
    elif treatment_type == RegressionTreatmentType.TARGET_ENCODE:
        df = Encoders.target_encode(df, cat_cols, target_col)
    
    else:
        raise ValueError("Invalid regression treatment type.")
    
    # Aplicar scaling solo si scaled=True
    if scaled and original_cont_cols:
        df = FeatureEngineer.scale_continuous(df, columns=None)
    
    return df