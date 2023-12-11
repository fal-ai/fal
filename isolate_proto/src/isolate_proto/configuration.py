# Defaults / descriptions https://github.com/grpc/grpc/blob/master/doc/keepalive.md

# Since the default timeout for the LB is 60 seconds, we can
# default to 20 seconds for the keepalive time.
from __future__ import annotations

GRPC_KEEPALIVE_TIME_MS = 30 * 1000

# Timeout when there's no activity for 10 minutes.
GRPC_KEEPALIVE_TIMEOUT_MS = 10 * 60 * 1000

# Currently, this is 2GiB, the max for a signed int.
GRPC_MAX_MESSAGE_SIZE = (2 * 1024 * 1024 * 1024) - 1

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_SIZE),
    ("grpc.keepalive_time_ms", GRPC_KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", GRPC_KEEPALIVE_TIMEOUT_MS),
    ("grpc.keepalive_permit_without_calls", 1),
    # Send an infinite number of pings
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_ping_interval_without_data_ms", GRPC_KEEPALIVE_TIME_MS - 5000),
    # Allow many strikes
    ("grpc.http2.max_ping_strikes", 0),
]
