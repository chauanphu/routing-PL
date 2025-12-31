from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from service import SolverService, SolverResult

app = FastAPI(title="VRP Solver API", description="API for Vehicle Routing Problem with Parcel Lockers")
service = SolverService()

class SolveRequest(BaseModel):
    instance_content: str
    solver: str = "paco"
    size: str = "small"
    params_override: Optional[Dict[str, Any]] = None

class RouteResult(BaseModel):
    objective: float
    vehicles: int
    runtime: float
    routes: List[List[str]]
    raw_routes: List[List[int]]
    delivery_nodes: List[int]
    success: bool
    error_message: str

@app.post("/solve", response_model=RouteResult)
async def solve(request: SolveRequest):
    """
    Run the VRP solver on the provided instance content.
    """
    result = service.run_solver(
        instance_content=request.instance_content,
        solver=request.solver,
        size=request.size,
        params_override=request.params_override
    )
    
    return result.to_dict()

@app.get("/health")
async def health():
    return {"status": "ok"}
