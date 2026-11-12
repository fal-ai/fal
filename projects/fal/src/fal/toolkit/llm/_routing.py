"""Replica routing state: which engine serves which request.

``_Router`` owns the replica list, session affinity, Responses API id
affinity and the replica chosen for the request in flight. Health probes and
engine process management stay with the app.
"""

import contextvars
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from fal.toolkit.llm._app import _Replica

# Session ids remembered for replica routing (least recently used dropped).
_SESSION_ROUTES_MAX = 4096
# Responses API ids remembered with the replica that created them, so a
# ``previous_response_id`` continuation reaches the engine holding its state.
_RESPONSE_ROUTES_MAX = 65536
# Extra requests in flight a session's replica may hold over the least loaded
# one before the session moves.
_STICKY_SLACK = 4


class _Router:
    def __init__(self, replicas: "List[_Replica]") -> None:
        self.replicas = replicas
        # Session id digest -> replica index, least recently used first.
        self._sessions: OrderedDict[bytes, int] = OrderedDict()
        # Responses API id -> index of the replica that created it.
        self._response_routes: OrderedDict[str, int] = OrderedDict()
        # The replica the current request was routed to: set when routing,
        # read where the engine call is made (tasks copy the context). Made
        # per instance, not at module level: a ContextVar cannot be pickled
        # with the class.
        self._current: contextvars.ContextVar[Optional[_Replica]] = (
            contextvars.ContextVar("llm_app_route", default=None)
        )

    def current(self) -> "_Replica":
        """The replica the current request was routed to. Outside a routed
        request (a subclass calling the engine directly) that is the first
        one."""
        return self._current.get() or self.replicas[0]

    def set_current(self, replica: "_Replica") -> None:
        self._current.set(replica)

    def pick(self, session: Optional[str]) -> "_Replica":
        """The replica that served ``session`` before, if it is still usable
        and not far busier than the rest; else the usable one with the fewest
        requests in flight."""
        replicas = self.replicas
        if len(replicas) == 1:
            return replicas[0]
        # With none usable the runner is failing its health check; any reply
        # (likely a 5xx) beats queueing here until it is replaced.
        usable = [r for r in replicas if r.usable()] or replicas
        choice = min(usable, key=lambda r: (r.inflight, r.index))
        key = None
        if session is not None:
            # A digest bounds the memory a long id can take.
            key = hashlib.blake2b(session.encode(), digest_size=16).digest()
            index = self._sessions.get(key)
            if index is not None and replicas[index] in usable:
                sticky = replicas[index]
                # A hot session id (say a constant ``user``) must not pile
                # every request onto one engine: past the least load plus the
                # larger of that load and the slack, it moves and loses its
                # cached prefix.
                slack = max(_STICKY_SLACK, choice.inflight)
                if sticky.inflight <= choice.inflight + slack:
                    self._sessions.move_to_end(key)
                    return sticky
        if key is not None:
            self._sessions[key] = choice.index
            self._sessions.move_to_end(key)
            while len(self._sessions) > _SESSION_ROUTES_MAX:
                self._sessions.popitem(last=False)
        return choice

    def continuation(self, path: str, payload: Dict[str, Any]) -> "Optional[_Replica]":
        """The replica that created ``previous_response_id``, if known. It is
        used even when busy or unhealthy: the engine keeps the conversation in
        memory, so any other replica would answer that the id is not found."""
        if path != "/v1/responses" or len(self.replicas) == 1:
            return None
        previous = payload.get("previous_response_id")
        if not isinstance(previous, str):
            return None
        index = self._response_routes.get(previous)
        if index is None:
            return None
        replica = self.replicas[index]
        if replica.proc is not None and replica.proc.poll() is not None:
            # The engine exited and its in-memory state went with it.
            del self._response_routes[previous]
            return None
        self._response_routes.move_to_end(previous)
        return replica

    def note_response(self, response_id: Any, replica: "_Replica") -> None:
        if not isinstance(response_id, str) or len(self.replicas) == 1:
            return
        self._response_routes[response_id] = replica.index
        self._response_routes.move_to_end(response_id)
        while len(self._response_routes) > _RESPONSE_ROUTES_MAX:
            self._response_routes.popitem(last=False)
