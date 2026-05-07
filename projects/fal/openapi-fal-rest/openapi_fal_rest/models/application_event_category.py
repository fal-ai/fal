from enum import Enum


class ApplicationEventCategory(str, Enum):
    CONFIG_CHANGED = "config_changed"
    DEPLOYMENT_ENDED = "deployment_ended"
    DEPLOYMENT_FAILED = "deployment_failed"
    DEPLOYMENT_RECREATE_APPLIED = "deployment_recreate_applied"
    DEPLOYMENT_ROLLING_ENDED = "deployment_rolling_ended"
    DEPLOYMENT_ROLLING_FAILED = "deployment_rolling_failed"
    DEPLOYMENT_ROLLING_STARTED = "deployment_rolling_started"
    DEPLOYMENT_STARTED = "deployment_started"
    RUNNER_DOCKER_PULL = "runner_docker_pull"
    RUNNER_DRAINING = "runner_draining"
    RUNNER_FAILED = "runner_failed"
    RUNNER_FINISHED = "runner_finished"
    RUNNER_PENDING = "runner_pending"
    RUNNER_SETUP = "runner_setup"
    RUNNER_STARTED = "runner_started"
    RUNNER_STARTUP_FAILURE = "runner_startup_failure"
    RUNNER_STOPPING = "runner_stopping"

    def __str__(self) -> str:
        return str(self.value)
