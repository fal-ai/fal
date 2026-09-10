# `fal dev`: reattachable development runners

## Status

Proposal and CLI prototype. The current fal platform does not have a first-class
`Devbox` resource or a runner port-forwarding API.

## Goal

Provide the smallest useful remote development loop as one command:

```console
$ fal dev
Creating devbox...
Attached to devbox 01H...

root@runner:/app#
```

After leaving the shell, running `fal dev` again from the same project should
reattach to the runner while it remains alive:

```console
$ fal dev
Reattaching to devbox 01H...
```

Inside the shell, developers can inspect the remote environment, edit files,
install Python or system dependencies, and run application code. This MVP does
not attempt to persist those changes into application configuration.

## MVP scope

The initial user-facing API is a single command:

```console
fal dev [--machine-type TYPE] [--idle-timeout SECONDS]
```

It performs three operations:

1. Find a live development runner previously associated with the current local
   project and attach to it.
2. If none exists, dispatch a long-running fal function.
3. Capture the allocated runner ID from the run's lifecycle log stream and
   attach using the existing `ShellRunner` gRPC stream.

The shell starts in `/app`. Source upload and synchronization are not part of
this initial increment.

Defaults:

- machine type: `S`
- runner keep-alive after the development workload is released: 7,200 seconds
- runner startup timeout: 15 minutes

The existing runner shell permission applies. On Windows, interactive runner
shells remain unsupported, matching `fal runners shell`.

## Reused platform capabilities

The prototype deliberately reuses existing APIs:

- normal fal run dispatch for runner allocation
- run lifecycle logs for initial runner discovery
- `ListRunners` for reattachment liveness checks
- `ShellRunner` for the interactive terminal
- runner `keep_alive` for a limited reattachment window

There is currently no dedicated API to create, identify, extend, or attach to a
devbox.

## Local project association

Until the platform has a first-class devbox identity, the CLI stores the selected
runner ID under:

```text
~/.fal/devboxes/<project-key>.json
```

The project key is derived from:

- the resolved current working directory
- the selected fal host
- the selected team

The state contains only the runner ID. Before reusing it, the CLI calls
`ListRunners` and verifies that the runner is in a shellable state. Missing,
invalid, and stale state is discarded and a new runner is created.

This is a prototype workaround, not the desired long-term contract. In
particular:

- reattachment is local-machine-specific
- moving or renaming the project creates a different association
- another workstation cannot discover the devbox
- concurrent creation attempts can race
- the backend does not know that the runner represents a devbox

## Runner creation and discovery

The CLI dispatches a function that remains active long enough for the shell to
attach. Run lifecycle logs include a bridge message in this form:

```text
Starting runner 4bf2e891-ede5-4d13-bf22-eba9621f9147
```

A custom result handler consumes the run's startup logs and captures the runner
ID from this message. When it receives the subsequent `Runner started`
lifecycle log, it makes the runner ID available to the shell attachment flow.
The CLI prints only the captured runner ID rather than the ordinary build and
runtime logs. This keeps the development entry flow concise and prevents
background output from corrupting the interactive terminal after it enters raw
mode. It also avoids calling `ListRunners` repeatedly during creation and
provides one place for future `fal dev`-specific log presentation.

Parsing a human-readable log remains a provisional contract. A structured
runner-start lifecycle event or an explicit create-devbox response containing
the runner ID would be more reliable.

## Non-goals for the MVP

The following are intentionally excluded:

- automatic source synchronization after creation
- bidirectional file synchronization
- app process start, stop, restart, or hot reload
- durable home or workspace volumes
- dependency freezing or configuration updates
- notebooks
- VS Code or Cursor remote connections
- port forwarding
- browser access or a new public hostname
- sharing devboxes between team members

Python and system package installation is possible interactively, but survives
only as long as the underlying runner does.

## Desired platform contract

If the workflow proves useful, the workaround should be replaced by a
server-owned resource. At minimum, the platform needs operations equivalent to:

```text
CreateDevbox(project_id, machine_type, idle_timeout, source) -> Devbox
GetDevbox(project_id) -> Devbox | not found
AttachShell(devbox_id) -> bidirectional terminal stream
DeleteDevbox(devbox_id)
```

A devbox should include at least:

```text
id
owner/team
project identity
runner id
state
machine type
created time
last attached time
idle expiration time
```

With that contract, `fal dev` would become straightforward:

1. Resolve a stable project identity.
2. Ask the service for an existing compatible devbox.
3. Create one if absent.
4. Attach to the returned devbox ID.

The service—not a local JSON file—would own identity, concurrency control,
liveness, expiry, authorization, and cross-machine reattachment.

## Possible follow-up increments

Each follow-up should build on a validated need rather than expanding the first
release:

1. `fal dev sync` or automatic one-way local-to-remote synchronization.
2. Managed `fal.App` start, restart, status, and logs.
3. Generic authenticated port forwarding.
4. JupyterLab through port forwarding.
5. An SSH-compatible bridge for VS Code and Cursor. `fal runners exec --interactive`
   already gives a raw byte stream when stdin is not a terminal, so
   `ProxyCommand fal runners exec -it <runner_id> -- /usr/sbin/sshd -i` works
   once `sshd` is installed in the devbox; a `fal dev` subcommand would only
   need to resolve the runner ID from the local state file.
6. Durable workspace and dependency persistence.

Notebooks and remote IDEs should share the same devbox lifecycle and filesystem;
they should not introduce separate compute resources.

## Open questions

- Is an idle runner guaranteed to retain the same writable filesystem for the
  full `keep_alive` period on every scheduler?
- Can `ShellRunner` attach safely to an idle runner, or only one with an active
  workload?
- Is the alias on `RunnerInfo` stable for ephemeral runs across all backends?
- Should detaching the shell immediately release the active run, or should the
  workload continue until its own deadline?
- Which project identity should eventually be stable across machines: app name,
  repository URL, explicit config ID, or a server-issued ID?
- Should installed packages and edits be ephemeral by definition, or should the
  workspace use a durable volume from the beginning?
