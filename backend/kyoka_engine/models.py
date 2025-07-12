from pydantic import BaseModel, Field
from typing import List

# Go CLI -> Python engine
class QueryRequest(BaseModel):
    query: str
    doc_paths: List[str]


# justification 
class Justification(BaseModel):
    clause: str
    reason: str

# Python engine -> Go CLI
class DecisionResponse(BaseModel):
    decision: str
    amount: str | None = None # optional 
    justification: List[Justification]


