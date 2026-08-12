FROM python:3.11-slim as build
RUN ln -s /usr/bin/python3 /tmp/python3
RUN python3 -m venv /opt/fal
COPY projects /src
RUN /opt/fal/bin/pip install /src/fal

FROM gcr.io/distroless/python3-debian12
COPY --from=build /tmp /usr/local/bin
COPY --from=build /opt/fal /opt/fal
ENTRYPOINT ["/opt/fal/bin/fal"]
