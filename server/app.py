"""
FastAPI server - exposes the environment over HTTP.
"""
import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import AgenticSecurityLabAction, AgenticSecurityLabObservation, AgenticSecurityLabState
from server.agentic_security_lab_environment import AgenticSecurityLabEnvironment

TASK_NAME = os.getenv("TASK_NAME", "easy")
DEFAULT_MODE = os.getenv("AGENTIC_SECURITY_LAB_MODE", "benchmark")
DEFAULT_FALLBACK = os.getenv("AGENTIC_SECURITY_LAB_COMMAND_FALLBACK", "false").lower() == "true"

# One shared environment instance per server process, protected by a lock so
# that concurrent HTTP requests (e.g. multiple users on the HF Space) cannot
# interleave their reads/writes on the same mutable episode state.
env = AgenticSecurityLabEnvironment(task_name=TASK_NAME)
_env_lock = threading.Lock()

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Agentic Security Lab — Supply Chain Incident Response",
    description=(
        "OpenEnv RL environment: respond to a live supply-chain attack. "
        "Quarantine malicious packages, rotate secrets, notify teams — "
        "before the attacker exfiltrates credentials."
    ),
    version="2.0.0",
)

# Allow cross-origin requests (needed for HF Spaces iframe embedding)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    with _env_lock:
        return env.reset(
            task_name=req.task_name,
            mode=req.mode or DEFAULT_MODE,
            command_fallback_enabled=(
                DEFAULT_FALLBACK if req.command_fallback_enabled is None else req.command_fallback_enabled
            ),
        )


@app.post("/step", response_model=AgenticSecurityLabObservation)
def step(req: StepRequest):
    action = AgenticSecurityLabAction(command=req.command, parameters=req.parameters)
    with _env_lock:
        return env.step(action)


@app.get("/state", response_model=AgenticSecurityLabState)
def state():
    with _env_lock:
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
        "modes": ["benchmark", "training"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "ui": "/web",
    }


# ── Frontend (served at /web for HF Spaces base_path) ───────────────────────

@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
@app.get("/web/index.html", include_in_schema=False)
def serve_ui():
    """Serve the interactive frontend dashboard."""
    index = _STATIC / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"error": "UI not found. Run from the project root."}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
