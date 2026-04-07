"""
FastAPI server — replace the contents of:
    agentic_security_lab/server/app.py
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

env = AgenticSecurityLabEnvironment(task_name=TASK_NAME)

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


class StepRequest(BaseModel):
    command: str
    parameters: dict = {}


@app.post("/reset", response_model=AgenticSecurityLabObservation)
def reset(req: ResetRequest = ResetRequest()):
    return env.reset(task_name=req.task_name)


@app.post("/step", response_model=AgenticSecurityLabObservation)
def step(req: StepRequest):
    action = AgenticSecurityLabAction(command=req.command, parameters=req.parameters)
    return env.step(action)


@app.get("/state", response_model=AgenticSecurityLabState)
def state():
    return env.state


@app.get("/health")
def health():
    return {"status": "ok", "task": env.state.task_name}

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