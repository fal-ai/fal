class ContainerImage:
    """ContainerImage represents a Docker image that can be built
    from a Dockerfile.
    """

    _known_keys = {"dockerfile_str", "build_env", "build_args"}

    @classmethod
    def from_dockerfile_str(cls, text: str, **kwargs):
        # Check for unknown keys and return them as a dict.
        return dict(
            dockerfile_str=text,
            **{k: v for k, v in kwargs.items() if k in cls._known_keys},
        )

    @classmethod
    def from_dockerfile(cls, path: str, **kwargs):
        with open(path) as fobj:
            return cls.from_dockerfile_str(fobj.read(), **kwargs)
