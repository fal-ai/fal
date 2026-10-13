from fal.toolkit.utils.download_utils import _filename_from_response
from fal.toolkit.utils.ssrf import SafeResponse


def test_filename_from_response_strips_directory_components():
    file_name = _filename_from_response(
        "https://example.com/file/with/slash.txt",
        SafeResponse(
            200,
            headers={
                "content-disposition": 'attachment; filename="file/with/slash.txt"'
            },
        ),
    )

    assert "/" not in file_name
