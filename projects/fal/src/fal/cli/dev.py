"""`fal dev`: personal GPU dev boxes (experimental). Lives in `fal.dev`."""


def add_parser(main_subparsers, parents):
    # Imported here: fal.dev.cli.parser imports fal.cli.parser, so a module-level
    # import would cycle through fal.cli.__init__ when fal.dev is imported first.
    from fal.dev.cli.parser import add_parser

    add_parser(main_subparsers, parents)
