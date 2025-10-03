from enum import Enum
from dataclasses import dataclass
from typing import Optional

class EncodingMethod(Enum):
    NONE = "none"       # ⭐⭐ NUEVO: No se aplica encoding
    LABEL = "label"
    ONEHOT = "onehot" 
    ORDINAL = "ordinal"
    TARGET = "target"
    FREQUENCY = "frequency"
    EMBED = "embed"
    BINARY = "binary"

@dataclass
class ColumnEncoding:
    """
    Configuración de encoding para una columna categórica específica.
    """
    column_name: str
    method: EncodingMethod
    params: Optional[dict] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    @classmethod
    def create(cls, column_name: str, method: EncodingMethod, **kwargs):
        """Método helper para crear configuraciones de encoding."""
        return cls(column_name, method, kwargs)