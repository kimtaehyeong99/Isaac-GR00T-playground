#!/bin/bash
set -e

IMAGE_NAME="gr00t-ffw"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$DIR/.."

echo "=== Isaac-GR00T-playground Docker Build ==="
echo "Project root: $PROJECT_ROOT"

# Ensure submodule is initialized
cd "$PROJECT_ROOT"
git submodule update --init

# Build with project root as context (so Dockerfile can access Isaac-GR00T/ and custom/)
export DOCKER_BUILDKIT=1
docker build \
    --platform linux/amd64 \
    --network host \
    -f "$DIR/Dockerfile" \
    "$@" \
    -t "$IMAGE_NAME" \
    "$PROJECT_ROOT" \
    && echo "=== Image $IMAGE_NAME BUILT SUCCESSFULLY ==="
