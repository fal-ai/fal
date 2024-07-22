PROJECT_TYPES = ["app"]


def _create_project(project_type: str):
    from cookiecutter.main import cookiecutter

    cookiecutter("https://github.com/fal-ai/cookiecutter-fal.git")


def add_parser(main_subparsers, parents):
    apps_help = "Create fal applications."
    parser = main_subparsers.add_parser(
        "create",
        description=apps_help,
        help=apps_help,
        parents=parents,
    )

    parser.add_argument(
        metavar="project_type",
        choices=PROJECT_TYPES,
        help="Type of project to create.",
        dest="project_type",
    )

    parser.set_defaults(func=_create_project)
