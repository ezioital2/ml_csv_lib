from abc import ABC, abstractmethod
import pandas as pd
from .treatments import SupervisedTreatmentType, UnsupervisedTreatmentType, RegressionTreatmentType, apply_supervised_treatment, apply_unsupervised_treatment, apply_regression_treatment
from .encoding_config import ColumnEncoding, EncodingMethod
from .encoders import Encoders
from .feature_engineer import FeatureEngineer
from sklearn.preprocessing import PowerTransformer

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
    def prepare_data(self, df: pd.DataFrame, treatment_type: RegressionTreatmentType, cat_cols: list, cont_cols: list, additional_transform: str = 'none', scaled: bool = True, **kwargs) -> tuple:
        """
        Prepares data for regression, with optional additional transformation (yeo-johnson, box-cox, none).
        
        :param df: DataFrame.
        :param treatment_type: RegressionTreatmentType enum.
        :param cat_cols: Categorical columns.
        :param cont_cols: Continuous columns.
        :param additional_transform: 'yeo-johnson', 'box-cox', or 'none'.
        :param scaled: Whether to scale continuous variables.
        :param kwargs: Must include 'target_col'.
        :return: X, y after treatment.
        """
        if 'target_col' not in kwargs:
            raise ValueError("Regression strategy requires 'target_col' in kwargs.")
        target_col = kwargs['target_col']
        
        # Hacer copia para no modificar las listas originales
        cat_cols = cat_cols.copy()
        cont_cols = cont_cols.copy()
        
        # Remove target from cat/cont if present
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        if target_col in cont_cols:
            cont_cols.remove(target_col)
        
        transformed_df = apply_regression_treatment(df, treatment_type, cat_cols, cont_cols, target_col, additional_transform, scaled)
        X, y = transformed_df.drop(target_col, axis=1), transformed_df[target_col]
        return X, y
class SupervisedPerColumnStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, column_encodings: list, cont_cols: list, scaled: bool = True, **kwargs) -> tuple:
        """
        Prepares data for supervised learning with per-column encoding.
        """
        if 'target_col' not in kwargs:
            raise ValueError("SupervisedPerColumnStrategy requires 'target_col' in kwargs.")
        target_col = kwargs['target_col']
        
        # Guardar target separadamente
        target_values = df[target_col].copy()
        
        # Remover target del DataFrame para transformaciones
        df_features = df.drop(target_col, axis=1)
        
        # Aplicar encoding por columna
        df_features = self._apply_per_column_encoding(df_features, column_encodings, target_col, target_values)
        
        # Aplicar scaling si es necesario
        if scaled:
            numeric_cols = df_features.select_dtypes(include=['int', 'float']).columns.tolist()
            if numeric_cols:
                df_features = FeatureEngineer.scale_continuous(df_features, columns=numeric_cols)
        
        # Reconstruir DataFrame con target
        df_features[target_col] = target_values
        
        X, y = df_features.drop(target_col, axis=1), df_features[target_col]
        return X, y
    
    def _apply_per_column_encoding(self, df: pd.DataFrame, column_encodings: list, target_col: str, target_values: pd.Series) -> pd.DataFrame:
        """Aplica encoding específico por columna."""
        df_encoded = df.copy()
        
        for col_encoding in column_encodings:
            col_name = col_encoding.column_name
            method = col_encoding.method
            params = col_encoding.params or {}
            
            if col_name not in df_encoded.columns:
                print(f"⚠️  Advertencia: Columna '{col_name}' no encontrada en DataFrame")
                continue
                
            # ⭐⭐ NUEVO: Manejar método NONE
            if method == EncodingMethod.NONE:
                # No se aplica encoding, se deja la columna como está
                print(f"🔸 Columna '{col_name}': Sin encoding (NONE)")
                continue
                
            elif method == EncodingMethod.LABEL:
                print(f"🔸 Columna '{col_name}': Label Encoding")
                df_encoded = Encoders.label_encode(df_encoded, [col_name])
                
            elif method == EncodingMethod.ONEHOT:
                print(f"🔸 Columna '{col_name}': One-Hot Encoding")
                df_encoded = Encoders.one_hot_encode(df_encoded, [col_name])
                
            elif method == EncodingMethod.ORDINAL:
                print(f"🔸 Columna '{col_name}': Ordinal Encoding")
                df_encoded = Encoders.ordinal_encode(df_encoded, [col_name])
                
            elif method == EncodingMethod.TARGET:
                print(f"🔸 Columna '{col_name}': Target Encoding")
                # Para target encoding necesitamos el target temporalmente
                df_temp = df_encoded.copy()
                df_temp[target_col] = target_values
                df_temp = Encoders.target_encode(df_temp, [col_name], target_col)
                df_encoded = df_temp.drop(target_col, axis=1)
                
            elif method == EncodingMethod.FREQUENCY:
                print(f"🔸 Columna '{col_name}': Frequency Encoding")
                df_encoded = Encoders.frequency_encode(df_encoded, [col_name])
                
            elif method == EncodingMethod.EMBED:
                embedding_dim = params.get('embedding_dim', 5)
                print(f"🔸 Columna '{col_name}': Embedding (dim={embedding_dim})")
                df_encoded = Encoders.embed_categorical(df_encoded, [col_name], embedding_dim)
                
            elif method == EncodingMethod.BINARY:
                print(f"🔸 Columna '{col_name}': Binary Encoding")
                # Binary encoding para alta cardinalidad
                codes = df_encoded[col_name].astype('category').cat.codes
                binary_repr = [list(map(int, list(format(x, '08b')[-8:]))) for x in codes]
                for i in range(8):
                    df_encoded[f"{col_name}_bin_{i}"] = [x[i] for x in binary_repr]
                df_encoded = df_encoded.drop(col_name, axis=1)
                
            else:
                raise ValueError(f"Método de encoding no soportado: {method}")
        
        return df_encoded

class UnsupervisedPerColumnStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, column_encodings: list, cont_cols: list, scaled: bool = True, **kwargs) -> pd.DataFrame:
        """
        Prepares data for unsupervised learning with per-column encoding.
        """
        df_encoded = df.copy()
        
        # Aplicar encoding por columna (sin target)
        for col_encoding in column_encodings:
            col_name = col_encoding.column_name
            method = col_encoding.method
            params = col_encoding.params or {}
            
            if col_name not in df_encoded.columns:
                print(f"⚠️  Advertencia: Columna '{col_name}' no encontrada en DataFrame")
                continue
                
            # ⭐⭐ NUEVO: Manejar método NONE
            if method == EncodingMethod.NONE:
                print(f"🔸 Columna '{col_name}': Sin encoding (NONE)")
                continue
                
            elif method == EncodingMethod.LABEL:
                print(f"🔸 Columna '{col_name}': Label Encoding")
                df_encoded = Encoders.label_encode(df_encoded, [col_name])
            elif method == EncodingMethod.ONEHOT:
                print(f"🔸 Columna '{col_name}': One-Hot Encoding")
                df_encoded = Encoders.one_hot_encode(df_encoded, [col_name])
            elif method == EncodingMethod.ORDINAL:
                print(f"🔸 Columna '{col_name}': Ordinal Encoding")
                df_encoded = Encoders.ordinal_encode(df_encoded, [col_name])
            elif method == EncodingMethod.FREQUENCY:
                print(f"🔸 Columna '{col_name}': Frequency Encoding")
                df_encoded = Encoders.frequency_encode(df_encoded, [col_name])
            elif method == EncodingMethod.EMBED:
                embedding_dim = params.get('embedding_dim', 5)
                print(f"🔸 Columna '{col_name}': Embedding (dim={embedding_dim})")
                df_encoded = Encoders.embed_categorical(df_encoded, [col_name], embedding_dim)
            elif method == EncodingMethod.BINARY:
                print(f"🔸 Columna '{col_name}': Binary Encoding")
                codes = df_encoded[col_name].astype('category').cat.codes
                binary_repr = [list(map(int, list(format(x, '08b')[-8:]))) for x in codes]
                for i in range(8):
                    df_encoded[f"{col_name}_bin_{i}"] = [x[i] for x in binary_repr]
                df_encoded = df_encoded.drop(col_name, axis=1)
            else:
                raise ValueError(f"Método de encoding no soportado para unsupervised: {method}")
        
        # Aplicar scaling si es necesario
        if scaled:
            numeric_cols = df_encoded.select_dtypes(include=['int', 'float']).columns.tolist()
            if numeric_cols:
                df_encoded = FeatureEngineer.scale_continuous(df_encoded, columns=numeric_cols)
        
        return df_encoded

class RegressionPerColumnStrategy(TreatmentStrategy):
    def prepare_data(self, df: pd.DataFrame, column_encodings: list, cont_cols: list, additional_transform: str = 'none', scaled: bool = True, **kwargs) -> tuple:
        """
        Prepares data for regression with per-column encoding.
        """
        if 'target_col' not in kwargs:
            raise ValueError("RegressionPerColumnStrategy requires 'target_col' in kwargs.")
        target_col = kwargs['target_col']
        
        # Guardar target separadamente
        target_values = df[target_col].copy()
        
        # Remover target del DataFrame para transformaciones
        df_features = df.drop(target_col, axis=1)
        
        # Aplicar transformación de potencia si se especifica
        if additional_transform != 'none' and cont_cols:
            pt = PowerTransformer(method=additional_transform, standardize=False)
            df_features[cont_cols] = pt.fit_transform(df_features[cont_cols])
        
        # Aplicar encoding por columna
        df_features = self._apply_per_column_encoding(df_features, column_encodings, target_col, target_values)
        
        # Aplicar scaling si es necesario
        if scaled:
            numeric_cols = df_features.select_dtypes(include=['int', 'float']).columns.tolist()
            if numeric_cols:
                df_features = FeatureEngineer.scale_continuous(df_features, columns=numeric_cols)
        
        # Reconstruir DataFrame con target
        df_features[target_col] = target_values
        
        X, y = df_features.drop(target_col, axis=1), df_features[target_col]
        return X, y
    
    def _apply_per_column_encoding(self, df: pd.DataFrame, column_encodings: list, target_col: str, target_values: pd.Series) -> pd.DataFrame:
        """Aplica encoding específico por columna para regresión."""
        df_encoded = df.copy()
        
        for col_encoding in column_encodings:
            col_name = col_encoding.column_name
            method = col_encoding.method
            params = col_encoding.params or {}
            
            if col_name not in df_encoded.columns:
                print(f"⚠️  Advertencia: Columna '{col_name}' no encontrada en DataFrame")
                continue
                
            # ⭐⭐ NUEVO: Manejar método NONE
            if method == EncodingMethod.NONE:
                print(f"🔸 Columna '{col_name}': Sin encoding (NONE)")
                continue
                
            elif method == EncodingMethod.LABEL:
                print(f"🔸 Columna '{col_name}': Label Encoding")
                df_encoded = Encoders.label_encode(df_encoded, [col_name])
            elif method == EncodingMethod.ONEHOT:
                print(f"🔸 Columna '{col_name}': One-Hot Encoding")
                df_encoded = Encoders.one_hot_encode(df_encoded, [col_name])
            elif method == EncodingMethod.ORDINAL:
                print(f"🔸 Columna '{col_name}': Ordinal Encoding")
                df_encoded = Encoders.ordinal_encode(df_encoded, [col_name])
            elif method == EncodingMethod.TARGET:
                print(f"🔸 Columna '{col_name}': Target Encoding")
                df_temp = df_encoded.copy()
                df_temp[target_col] = target_values
                df_temp = Encoders.target_encode(df_temp, [col_name], target_col)
                df_encoded = df_temp.drop(target_col, axis=1)
            elif method == EncodingMethod.FREQUENCY:
                print(f"🔸 Columna '{col_name}': Frequency Encoding")
                df_encoded = Encoders.frequency_encode(df_encoded, [col_name])
            elif method == EncodingMethod.EMBED:
                embedding_dim = params.get('embedding_dim', 5)
                print(f"🔸 Columna '{col_name}': Embedding (dim={embedding_dim})")
                df_encoded = Encoders.embed_categorical(df_encoded, [col_name], embedding_dim)
            elif method == EncodingMethod.BINARY:
                print(f"🔸 Columna '{col_name}': Binary Encoding")
                codes = df_encoded[col_name].astype('category').cat.codes
                binary_repr = [list(map(int, list(format(x, '08b')[-8:]))) for x in codes]
                for i in range(8):
                    df_encoded[f"{col_name}_bin_{i}"] = [x[i] for x in binary_repr]
                df_encoded = df_encoded.drop(col_name, axis=1)
            else:
                raise ValueError(f"Método de encoding no soportado: {method}")
        
        return df_encoded