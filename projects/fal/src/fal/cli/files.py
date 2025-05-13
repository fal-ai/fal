from .parser import FalClientParser


def _list(args):
    import posixpath

    from fal.files import FalFileSystem

    fs = FalFileSystem()

    for entry in fs.ls(args.path, detail=True):
        name = posixpath.basename(entry["name"])
        color = "blue" if entry["type"] == "directory" else "default"
        args.console.print(f"[{color}]{name}[/{color}]")


def _download(args):
    from fal.files import FalFileSystem

    fs = FalFileSystem()
    fs.get(args.remote_path, args.local_path)


def _upload(args):
    from fal.files import FalFileSystem

    fs = FalFileSystem()
    fs.put(args.local_path, args.remote_path)


def add_parser(main_subparsers, parents):
    files_help = "Manage fal files."
    parser = main_subparsers.add_parser(
        "files",
        aliases=["file"],
        description=files_help,
        help=files_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    list_parser = subparsers.add_parser("list", aliases=["ls"], parents=parents)
    list_parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help="The path to list",
        default="/",
    )
    list_parser.set_defaults(func=_list)

    download_parser = subparsers.add_parser("download", parents=parents)
    download_parser.add_argument(
        "remote_path", type=str, help="Remote path to download"
    )
    download_parser.add_argument(
        "local_path", type=str, help="Local path to download to"
    )
    download_parser.set_defaults(func=_download)

    upload_parser = subparsers.add_parser("upload", parents=parents)
    upload_parser.add_argument("local_path", type=str, help="Local path to upload")
    upload_parser.add_argument("remote_path", type=str, help="Remote path to upload to")
    upload_parser.set_defaults(func=_upload)
