Node-local uploads (internal opt-in)
===================================

Selected runners can send SDK outputs to the node-local uploader instead of
waiting for the remote storage transfer. Set these variables in the runner's
environment, with an SDK version containing this feature::

    FAL_USE_LOCAL_UPLOADER=1
    CDN_UPLOADER_SOCKET_PATH=/run/fal-upload/upload.sock

The socket path is optional and defaults to the value above. The uploader service
and its socket-directory mount must already be provisioned on selected nodes
(INFRA-5241). Setting the SDK flag does not deploy the service or mount the socket.
Served apps already include HTTPX. Plain-function environments that opt in must
also include HTTPX; disabled runners do not acquire a new load-time dependency.

Existing ``File.from_bytes``, ``File.from_path``, their async wrappers, and
``Image`` uploads use the new path when their repository is the default
``fal_v3`` (including its deprecated ``fal_v2`` and ``cdn`` aliases). Explicit
repository objects, other repositories, and caller-supplied upload policies keep
their existing behavior. ``fal_client`` is unaffected.

The flag is read for each upload on the runner, including for serialized apps.
Absent, empty, ``0``, or ``false`` disables it. Socket presence alone does not
enable it. Roll back by disabling the flag on replacement runners; keep the
uploader running to finish already accepted work. Start with the staging canary
in INFRA-5242 before enabling additional runners.

Acceptance and failures
-----------------------

The SDK returns only after ``202 accepted_local``: the entire file is durable on
the node. Remote transfer continues independently and survives caller exit.
This does not promise immediate CDN availability or survival of spool-disk loss.
Callers must not modify a source file until the upload call returns. Large files
stream from an open handle. As before, small files retain ``file_data`` for
``as_bytes()``; ``multipart=True`` disables that retention. Remote multipart
thresholds, part sizes, and concurrency are controlled by the uploader, not the
SDK's multipart tuning arguments.

The adapter forwards the request's CDN token and request ID, available REST
credentials, content type, filename, and effective lifecycle settings. The
uploader decides whether to reserve directly at CDN or use REST for account
defaults and recipient lookup. Missing settings remain absent.

Local failures do not use the legacy repository fallback or automatically retry.
``LocalUploadError`` is a ``FileUploadException`` with ``status_code`` when an HTTP
error was received and ``acceptance_uncertain`` when work may have been accepted.
Missing sockets, full queues, and interrupted receives are errors. Losing the
acceptance response does not undo acceptance; resubmitting may create another
URL, and the request ID is not an idempotency key.

The existing async wrappers run synchronous uploads in worker threads. Cancelling
the await does not stop that thread or undo acceptance. The worker retains its
source handle and does not start another submission. Socket operations have a
five-second connect timeout and a 300-second inactivity timeout, not an overall
generation deadline.

Generated streams
-----------------

The internal ``fal.toolkit.file._local_uploader.LocalUploader`` client also
supports incremental producers. The caller supplies credentials and settings
using the uploader's headers; use a session context manager to abort unfinished
work on exit::

    from fal.toolkit.file._local_uploader import LocalUploader

    headers = {
        "Authorization": "Key <caller-key>",
        "Content-Type": "video/mp4",
    }
    size = 0

    def body():
        global size
        for chunk in generate_video():
            size += len(chunk)
            yield chunk

    with LocalUploader() as client:
        with client.begin_stream("video.mp4", headers) as session:
            session.send_body(body())
            accepted = session.finish(size)
    # accepted.upload_id is the durable upload ID, not the old session ID.
    url = accepted.file_url

Body EOF and the reserved session URL do not imply acceptance. Only call
``finish`` after successful generation and body reception, using the exact final
byte count. Empty streams cannot finish; use an ordinary zero-length upload.
Producer failures attempt abort without masking the original error. Abort cannot
undo a finish already being committed.

Local validation
----------------

Unit and Unix-socket tests require no remote service::

    pytest projects/fal/tests/unit/toolkit/file/test_local_uploader*.py
    pytest projects/fal/tests/unit/test_serialization.py

To test the real Rust uploader with a controlled local HTTP storage peer, build
``cdn-uploader`` in isolate-cloud and run::

    CDN_UPLOADER_TEST_BINARY=/path/to/cdn-uploader \
      pytest projects/fal/tests/integration/toolkit/test_local_uploader.py

These process tests use disposable local credentials. They cover ordinary and
multipart files, caller exit before remote completion, queue exhaustion, and
explicit stream finish. They do not replace the staging/real-CDN canary.
