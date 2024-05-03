class ContainerImage:
    """ContainerImage represents a Docker image that can be built
    from a Dockerfile.
    """

    @classmethod
    def from_dockerfile_str(cls, text: str, **kwargs):
        return dict(dockerfile_str=text, **kwargs)

    @classmethod
    def from_dockerfile(cls, path: str, **kwargs):
        with open(path) as fobj:
            return cls.from_dockerfile_str(fobj.read(), **kwargs)
