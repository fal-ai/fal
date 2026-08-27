from types import SimpleNamespace
from unittest.mock import Mock

from fal.files import FalFileSystem


def test_multipart_reports_checksum_then_upload(monkeypatch, tmp_path):
    path = tmp_path / "model.bin"
    progress = Mock()
    progress.add_task.return_value = 7

    def compute_md5(lpath):
        assert lpath == str(path)
        progress.add_task.assert_called_once_with("Calculating checksum...", total=1)
        progress.update.assert_not_called()
        return "etag"

    uploader = Mock()

    def upload_file(lpath, on_part_complete):
        assert lpath == str(path)
        progress.update.assert_called_once_with(7, description="Uploading model.bin")
        return "etag"

    uploader.upload_file.side_effect = upload_file
    monkeypatch.setattr("fal.files._compute_md5", compute_md5)
    monkeypatch.setattr(
        "fal.files.DataFileMultipartUpload", Mock(return_value=uploader)
    )

    FalFileSystem._put_file_multipart(
        SimpleNamespace(_client=object()), str(path), "/data/model.bin", 1, progress
    )

    uploader.upload_file.assert_called_once()
