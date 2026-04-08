"""
FastAPI server — exposes the environment over HTTP.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel

from models import (
    AgenticSecurityLabAction,
    AgenticSecurityLabObservation,
    AgenticSecurityLabState,
)
from server.agentic_security_lab_environment import AgenticSecurityLabEnvironment

TASK_NAME = os.getenv("TASK_NAME", "easy")
MODE = os.getenv("AGENTIC_SECURITY_LAB_MODE", "benchmark")
COMMAND_FALLBACK = os.getenv("AGENTIC_SECURITY_LAB_COMMAND_FALLBACK", "true").lower() in {"1", "true", "yes"}

env = AgenticSecurityLabEnvironment(
    task_name=TASK_NAME,
    mode=MODE,
    command_fallback_enabled=COMMAND_FALLBACK,
)

app = FastAPI(
    title="Agentic Security Lab — Supply Chain Incident Response",
    description=(
        "OpenEnv RL environment: respond to a live supply-chain attack. "
        "Quarantine malicious packages, rotate secrets, notify teams — "
        "before the attacker exfiltrates credentials."
    ),
    version="1.0.0",
)


class ResetRequest(BaseModel):
    task_name: str | None = None
    mode: str | None = None
    command_fallback_enabled: bool | None = None


class StepRequest(BaseModel):
    command: str
    parameters: dict = {}


@app.post("/reset", response_model=AgenticSecurityLabObservation)
def reset(req: ResetRequest = ResetRequest()):
    return env.reset(
        task_name=req.task_name,
        mode=req.mode,
        command_fallback_enabled=req.command_fallback_enabled,
    )


@app.post("/step", response_model=AgenticSecurityLabObservation)
def step(req: StepRequest):
    action = AgenticSecurityLabAction(command=req.command, parameters=req.parameters)
    return env.step(action)


@app.get("/state", response_model=AgenticSecurityLabState)
def state():
    return env.state


@app.get("/health")
def health():
    return {
        "status": "ok",
        "task": env.state.task_name,
        "mode": env.state.mode,
        "mode_fallback_used": env.state.mode_fallback_used,
    }


@app.get("/")
def root():
    return {
        "name": "agentic-security-lab",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()