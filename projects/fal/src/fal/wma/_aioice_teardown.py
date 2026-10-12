"""Orderly ICE teardown compatibility shim for ``aioice``.

Make :meth:`aioice.ice.Connection.close` cancel — and *await* — its in-flight
ICE connectivity-check tasks before it closes the underlying UDP transports, so
no orphaned STUN retransmission timer can fire against an already-closed socket.

Root cause (aioice==0.10.2, and upstream ``main`` as of 2026-07-27 — no released
or unreleased fix exists)
-------------------------------------------------------------------------------
``Connection.close()`` tears a connection down by signalling ``ICE_FAILED`` and
then closing each ``StunProtocol``'s datagram transport. It never cancels the
per-candidate-pair check tasks (``CandidatePair.task``). ``Connection.connect()``
*does* cancel them, but (a) it does not await the cancellation and (b) *triggered*
checks spawned from late-arriving remote STUN binding requests
(``check_incoming`` -> ``check_start_task``) create new check tasks *after*
connect()'s one-shot cancel sweep. Any such task is still awaiting a
``stun.Transaction`` whose retransmission timer is a live
``loop.call_later(delay, Transaction.__retry)`` (``RETRY_RTO`` 0.5s, doubling, up
to ``RETRY_MAX`` = 6 retries -> ~31s of scheduled retries).

When ``close()`` closes the transport first, the next ``__retry`` calls
``send_stun`` -> ``transport.sendto`` on a ``_SelectorDatagramTransport`` whose
``_sock`` is already ``None`` (asyncio's ``_conn_lost`` fast-return is skipped
because ICE host-candidate transports are created without a ``remote_addr``, so
``_address`` is unset), raising an ``AttributeError`` that asyncio logs as
``Exception in callback Transaction.__retry()``. On a session whose first WebRTC
attempt failed and was torn down, these orphaned timers keep firing for seconds
*into the next (successful) attempt* — the "first attempt fails, second works,
loop logs Transaction.__retry errors during normal operation" report. It is pure
teardown log noise (it only ever touches the dead connection's own objects; the
next session uses fresh protocols/transactions), but it is real and avoidable.

The fix mirrors what ``connect()`` already does, but at the correct point and
*with* an await: cancel every check task and await it so its
``Transaction.run()`` ``finally`` cancels the retransmission timer *before* the
transports are closed. The original ``close()`` then runs unchanged.

Design constraints honoured
---------------------------
- No global ``asyncio`` exception handler and no broad "swallow every callback
  error" behaviour: we only await tasks we are actively cancelling during
  teardown, which is exactly what ``connect()`` already does with them.
- Idempotent: installing twice is a no-op.
- Version-guarded and self-disabling: if a future ``aioice`` teaches ``close()``
  to cancel check tasks itself, the source guard below detects it and declines
  to wrap, so we never double-cancel. ``tests/unit/wma/test_wma_helpers.py`` pins both
  the unpatched race and the patched fix, and will flag such an upstream change.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

#: Sentinel set on our wrapper so a second install is a no-op (idempotence).
_MARKER = "_fal_wma_orderly_teardown"


def _close_already_cancels_check_tasks(close_fn: object) -> bool:
    """True if this ``Connection.close`` already cancels check-list tasks.

    Used as a self-disabling upstream-fix guard: if a newer aioice teaches
    ``close()`` to tear down ``CandidatePair.task`` itself, we must not wrap it
    (that would double-cancel). Best-effort — if the source can't be read we
    assume the unpatched shape (the only shape that has ever shipped).
    """
    try:
        src = inspect.getsource(close_fn)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return False
    return ".task" in src and "cancel" in src


def install_orderly_ice_teardown() -> bool:
    """Idempotently patch :meth:`aioice.ice.Connection.close` for orderly teardown.

    Returns ``True`` if the patch is now installed (or was already), ``False`` if
    it was intentionally skipped (upstream already handles it, or aioice absent).
    Safe to call once per session before any connection is closed; the apps call
    it from :func:`fal.wma._raw.negotiate_answer`, which runs before any ICE
    teardown can occur.
    """
    try:
        from aioice import ice
    except Exception:  # pragma: no cover - aioice is a runner-only package
        return False

    close_fn = ice.Connection.close
    if getattr(close_fn, _MARKER, False):
        return True  # already installed
    if _close_already_cancels_check_tasks(close_fn):
        logger.debug(
            "aioice Connection.close already cancels check tasks; "
            "skipping fal.wma orderly-teardown shim"
        )
        return False

    original_close = close_fn

    async def _orderly_close(self) -> None:  # type: ignore[no-untyped-def]
        # Cancel every in-flight ICE check task, then await it so its
        # stun.Transaction.run() finally cancels the retransmission timer,
        # BEFORE the original close() closes the transports out from under it.
        check_list = getattr(self, "_check_list", None) or []
        pending = [
            pair.task
            for pair in list(check_list)
            if getattr(pair, "task", None) is not None and not pair.task.done()
        ]
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except BaseException:  # noqa: BLE001 - teardown of a task we cancelled
                # A cancelled/failed check task's only remaining side effect is
                # its (now-cancelled) retransmit timer; its outcome is irrelevant
                # to teardown, exactly as connect() treats these tasks.
                pass
        await original_close(self)

    setattr(_orderly_close, _MARKER, True)
    ice.Connection.close = _orderly_close  # type: ignore[method-assign]
    logger.debug("installed fal.wma orderly ICE teardown shim on aioice")
    return True
