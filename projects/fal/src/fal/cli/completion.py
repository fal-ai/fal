from __future__ import annotations

import argcomplete

SHELLS = ("bash", "zsh", "tcsh", "fish")


def _completion(args) -> int:
    print(
        argcomplete.shellcode(
            ["fal"],
            use_defaults=True,
            shell=args.shell,
        ),
        end="",
    )
    return 0


def add_parser(main_subparsers, parents):
    completion_help = "Generate shell completion scripts."
    description = (
        "Generate shell completion scripts for the fal CLI.\n\n"
        "The command prints shell code to stdout. To enable completion, "
        "add one of the following to your shell configuration and restart "
        "your shell (or source the file):\n\n"
        "  Bash/Zsh:\n"
        '    eval "$(fal completion bash)"\n'
        '    eval "$(fal completion zsh)"\n\n'
        "  Fish:\n"
        "    fal completion fish > ~/.config/fish/completions/fal.fish\n\n"
        "  Tcsh:\n"
        "    fal completion tcsh >> ~/.tcshrc"
    )
    epilog = (
        "Examples:\n"
        '  eval "$(fal completion bash)"\n'
        "  fal completion fish > ~/.config/fish/completions/fal.fish\n"
    )
    parser = main_subparsers.add_parser(
        "completion",
        description=description,
        help=completion_help,
        epilog=epilog,
        parents=parents,
    )
    parser.add_argument(
        "shell",
        choices=SHELLS,
        help="Shell to generate a completion script for.",
    )
    parser.set_defaults(func=_completion)
