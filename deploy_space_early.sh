#!/usr/bin/env bash
set -euo pipefail

echo "Validating OpenEnv manifest..."
openenv validate .

echo "Pushing early environment build to Hugging Face Space..."
openenv push

echo "Done. Share the Space URL with teammates for integration tests."
