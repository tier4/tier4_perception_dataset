#!/usr/bin/env bash
set -e

IMAGE_NAME="tier4-perception-converter"
VERSION=$(grep '^version' pyproject.toml | cut -d'"' -f2)

echo "Building ${IMAGE_NAME}:${VERSION}..."

docker build -t "${IMAGE_NAME}:${VERSION}" -t "${IMAGE_NAME}:latest" .

echo ""
echo "Build complete!"
echo "  Image: ${IMAGE_NAME}:${VERSION}"
echo "  Image: ${IMAGE_NAME}:latest"