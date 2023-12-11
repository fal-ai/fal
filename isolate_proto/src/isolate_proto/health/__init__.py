from __future__ import annotations

from isolate_proto.health.health_pb2 import *
from isolate_proto.health.health_pb2_grpc import HealthServicer, HealthStub
from isolate_proto.health.health_pb2_grpc import (
    add_HealthServicer_to_server as register_health,
)
