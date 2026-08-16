#!/usr/bin/env bash
set -euo pipefail

uv run reviews-download --bytes 16777216
uv run reviews-transform
uv run reviews-model
uv run reviews-evidence

