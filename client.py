# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agentic Security Lab Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AgenticSecurityLabAction, AgenticSecurityLabObservation


class AgenticSecurityLabEnv(
    EnvClient[AgenticSecurityLabAction, AgenticSecurityLabObservation, State]
):
    """
    Client for the Agentic Security Lab Environment.

    This client sends structured incident-response actions:
      - command: str
      - parameters: dict
    """

    def _step_payload(self, action: AgenticSecurityLabAction) -> Dict:
        return {
            "command": action.command,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AgenticSecurityLabObservation]:
        obs_data = payload.get("observation", payload)
        observation = AgenticSecurityLabObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
