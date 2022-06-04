from pydantic import BaseModel,Json
from typing import List, Optional,Dict


class Message(BaseModel):
    text : str

class Switch(BaseModel):
    state : bool