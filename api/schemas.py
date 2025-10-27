# api/schemas.py
from pydantic import BaseModel
from typing import Dict

class PWaveFeatures(BaseModel):
    __root__: Dict[str, float]
