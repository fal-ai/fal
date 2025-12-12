from typing import TYPE_CHECKING

from .parser import FalClientParser

if TYPE_CHECKING:
    from fal.files import FalFileSystem


def _get_fs(args) -> "FalFileSystem":
    from fal.files import FalFileSystem

    return FalFileSystem(host=args.host, team=args.team)


def _list(args):
    import posixpath

    for entry in _get_fs(args).ls(args.path, detail=True):
        name = posixpath.basename(entry["name"])
        color = "blue" if entry["type"] == "directory" else "default"
        args.console.print(f"[{color}]{name}[/{color}]")


def _download(args):
    _get_fs(args).get(args.remote_path, args.local_path, recursive=True)


def _upload(args):
    _get_fs(args).put(args.local_path, args.remote_path, recursive=True)


def _upload_url(args):
    _get_fs(args).put_file_from_url(args.url, args.remote_path)


def _mv(args):
    _get_fs(args).mv(args.source, args.destination)


def _rm(args):
    _get_fs(args).rm(args.path)


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
        dest="cmd",
        required=True,
        parser_class=FalClientParser,
    )

    list_help = "List files."
    list_parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        description=list_help,
        help=list_help,
        parents=parents,
    )
    list_parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help="The path to list",
        default="/",
    )
    list_parser.set_defaults(func=_list)

    download_help = "Download files."
    download_parser = subparsers.add_parser(
        "download",
        description=download_help,
        help=download_help,
        parents=parents,
    )
    download_parser.add_argument(
        "remote_path", type=str, help="Remote path to download"
    )
    download_parser.add_argument(
        "local_path", type=str, help="Local path to download to"
    )
    download_parser.set_defaults(func=_download)

    upload_help = "Upload files."
    upload_parser = subparsers.add_parser(
        "upload",
        description=upload_help,
        help=upload_help,
        parents=parents,
    )
    upload_parser.add_argument("local_path", type=str, help="Local path to upload")
    upload_parser.add_argument("remote_path", type=str, help="Remote path to upload to")
    upload_parser.set_defaults(func=_upload)

    upload_url_help = "Upload file from URL."
    upload_url_parser = subparsers.add_parser(
        "upload-url",
        description=upload_url_help,
        help=upload_url_help,
        parents=parents,
    )
    upload_url_parser.add_argument("url", type=str, help="URL to upload")
    upload_url_parser.add_argument(
        "remote_path", type=str, help="Remote path to upload to"
    )
    upload_url_parser.set_defaults(func=_upload_url)

    mv_help = "Move or rename a remote file or directory."
    mv_parser = subparsers.add_parser(
        "mv",
        description=mv_help,
        help=mv_help,
        parents=parents,
    )
    mv_parser.add_argument("source", type=str, help="Remote source path")
    mv_parser.add_argument("destination", type=str, help="Remote destination path")
    mv_parser.set_defaults(func=_mv)

    rm_help = "Recursively remove a remote file or directory."
    rm_parser = subparsers.add_parser(
        "rm",
        description=rm_help,
        help=rm_help,
        parents=parents,
    )
    rm_parser.add_argument("path", type=str, help="Remote path")
    rm_parser.set_defaults(func=_rm)
