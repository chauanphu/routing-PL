"""
Solver Runner Module

Invokes the compiled C++ VRP solver and parses the JSON output.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SolverResult:
    """Result from running the VRP solver."""
    objective: float
    vehicles: int
    runtime: float
    routes: List[List[int]]
    delivery_nodes: List[int] = field(default_factory=list)
    success: bool = True
    error_message: str = ""


# Available solvers
AVAILABLE_SOLVERS = ["paco", "sa"]

# Size options
SIZE_OPTIONS = ["small", "medium", "large"]


def get_solver_binary_path() -> Path:
    """Get path to the solver binary."""
    return Path(__file__).parent / "bin" / "test"


def get_params_path(solver: str) -> Path:
    """Get path to solver parameter file."""
    return Path(__file__).parent / "parameters" / f"{solver}.param.yaml"


def run_solver(
    instance_content: str,
    solver: str = "paco",
    size: str = "small",
) -> SolverResult:
    """
    Run the VRP solver on an instance file.
    
    Args:
        instance_content: Content of the VRP instance file
        solver: Solver name ("paco" or "sa")
        size: Problem size ("small", "medium", "large")
        
    Returns:
        SolverResult with solution data or error information
    """
    if solver not in AVAILABLE_SOLVERS:
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message=f"Unknown solver: {solver}. Available: {AVAILABLE_SOLVERS}"
        )
    
    if size not in SIZE_OPTIONS:
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message=f"Unknown size: {size}. Available: {SIZE_OPTIONS}"
        )
    
    solver_path = get_solver_binary_path()
    params_path = get_params_path(solver)
    
    if not solver_path.exists():
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message=f"Solver binary not found at: {solver_path}"
        )
    
    if not params_path.exists():
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message=f"Parameter file not found at: {params_path}"
        )
    
    # Write instance to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(instance_content)
        instance_path = Path(f.name)
    
    try:
        # Build command
        cmd = [
            str(solver_path),
            "--solver", solver,
            "--params", str(params_path),
            "--instance-file", str(instance_path),
            "--size", size,
            "--full-solution"
        ]
        
        # Run solver
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return SolverResult(
                objective=0,
                vehicles=0,
                runtime=0,
                routes=[],
                success=False,
                error_message=f"Solver exited with code {result.returncode}: {result.stderr}"
            )
        
        # Parse JSON output
        # Find the JSON part in stdout (after any log messages)
        stdout = result.stdout
        json_start = stdout.find('{')
        if json_start == -1:
            return SolverResult(
                objective=0,
                vehicles=0,
                runtime=0,
                routes=[],
                success=False,
                error_message=f"No JSON output found in solver output: {stdout[:500]}"
            )
        
        json_str = stdout[json_start:]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return SolverResult(
                objective=0,
                vehicles=0,
                runtime=0,
                routes=[],
                success=False,
                error_message=f"Failed to parse JSON: {e}. Output: {json_str[:500]}"
            )
        
        return SolverResult(
            objective=data.get("objective", 0),
            vehicles=data.get("vehicles", 0),
            runtime=data.get("runtime", 0),
            routes=data.get("routes", []),
            delivery_nodes=data.get("delivery_nodes", []),
            success=True,
        )
        
    except subprocess.TimeoutExpired:
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message="Solver timed out after 5 minutes"
        )
    except Exception as e:
        return SolverResult(
            objective=0,
            vehicles=0,
            runtime=0,
            routes=[],
            success=False,
            error_message=f"Error running solver: {e}"
        )
    finally:
        # Clean up temp file
        try:
            instance_path.unlink()
        except:
            pass
