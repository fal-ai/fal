# Registry the Docker Hub base image is pulled from. CI overrides this with a
# pull-through cache so builds do not draw on the anonymous docker.io rate
# limit; unset it and the build behaves exactly as before.
ARG DOCKER_MIRROR=docker.io

FROM ${DOCKER_MIRROR}/library/python:3.11-slim as build
RUN ln -s /usr/bin/python3 /tmp/python3
RUN python3 -m venv /opt/fal
COPY projects /src
RUN /opt/fal/bin/pip install /src/fal

FROM gcr.io/distroless/python3-debian12
COPY --from=build /tmp /usr/local/bin
COPY --from=build /opt/fal /opt/fal
ENTRYPOINT ["/opt/fal/bin/fal"]
