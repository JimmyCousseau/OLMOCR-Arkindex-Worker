#!/bin/sh -e
# Build the tasks Docker image.
# Requires CI_PROJECT_DIR and CI_REGISTRY_IMAGE to be set.
# Will automatically login to a registry if CI_REGISTRY, CI_REGISTRY_USER and CI_REGISTRY_PASSWORD are set.
# Will only push an image if $CI_REGISTRY is set.

if [ -z "$VERSION" ] || [ -z "$CI_PROJECT_DIR" ] || [ -z "$CI_REGISTRY_IMAGE" ]; then
  echo Missing environment variables
  exit 1
fi

IMAGE_TAG="$CI_REGISTRY_IMAGE:$VERSION"

cd "$CI_PROJECT_DIR"
docker build -f Dockerfile . -t "$IMAGE_TAG"

if [ -n "$CI_REGISTRY" ] && [ -n "$CI_REGISTRY_USER" ] && [ -n "$CI_REGISTRY_PASSWORD" ]; then
  echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
  docker push "$IMAGE_TAG"
else
  echo "Missing environment variables to log in to the container registry…"
fi
