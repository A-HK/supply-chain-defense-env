# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agentic Security Lab Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AgenticSecurityLabAction, AgenticSecurityLabObservation


class AgenticSecurityLabEnv(
    EnvClient[AgenticSecurityLabAction, AgenticSecurityLabObservation, State]
):
    """Client for the Agentic Security Lab environment."""

    def _step_payload(self, action: AgenticSecurityLabAction) -> Dict:
        """
        Convert AgenticSecurityLabAction to JSON payload for step message.

        Args:
            action: AgenticSecurityLabAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "command": action.command,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AgenticSecurityLabObservation]:
        """
        Parse server response into StepResult[AgenticSecurityLabObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AgenticSecurityLabObservation
        """
        obs_data = payload.get("observation", {})
        observation = AgenticSecurityLabObservation(
            success=obs_data.get("success", payload.get("success", True)),
            done=payload.get("done", False),
            reward=float(payload.get("reward", 0.0)),
            result=obs_data.get("result", payload.get("result", "")),
            data=obs_data.get("data", payload.get("data", {})),
            incident_summary=obs_data.get("incident_summary", payload.get("incident_summary", "")),
            steps_remaining=obs_data.get("steps_remaining", payload.get("steps_remaining", 0)),
            exposed_secrets=obs_data.get("exposed_secrets", payload.get("exposed_secrets", [])),
            active_malicious_packages=obs_data.get(
                "active_malicious_packages",
                payload.get("active_malicious_packages", []),
            ),
            visible_alerts=obs_data.get("visible_alerts", payload.get("visible_alerts", [])),
            uncertainty_score=float(obs_data.get("uncertainty_score", payload.get("uncertainty_score", 0.0))),
            plan_progress=obs_data.get("plan_progress", payload.get("plan_progress", {})),
            info=obs_data.get("info", payload.get("info", {})),
            error=obs_data.get("error", payload.get("error")),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
