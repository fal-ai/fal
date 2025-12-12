import posixpath
import uuid

from fal.files import FalFileSystem


def test_fal_fs(tmp_path):
    (tmp_path / "myfile").write_text("myfile")
    (tmp_path / "mydir").mkdir()
    (tmp_path / "mydir" / "myfile1").write_text("myfile1")
    (tmp_path / "mydir" / "myfile2").write_text("myfile2")

    (tmp_path / "downloaded").mkdir()

    remote_temp_dir = f"/data/tmp/{uuid.uuid4()}"

    fs = FalFileSystem()

    assert fs.exists("/")
    assert fs.isdir("/")
    assert fs.exists("/data")
    assert fs.isdir("/data")
    assert fs.exists(".")
    assert fs.isdir(".")

    fs.put(str(tmp_path / "myfile"), remote_temp_dir + "/myfile")
    assert fs.exists(remote_temp_dir + "/myfile")
    assert fs.isdir(remote_temp_dir)
    assert fs.isfile(remote_temp_dir + "/myfile")
    assert not fs.isdir(remote_temp_dir + "/myfile")
    fs.get(remote_temp_dir + "/myfile", str(tmp_path / "downloaded" / "myfile"))
    assert (tmp_path / "downloaded" / "myfile").read_text() == "myfile"

    fs.put(str(tmp_path / "mydir"), remote_temp_dir + "/mydir", recursive=True)
    assert fs.exists(remote_temp_dir + "/mydir")
    assert fs.isdir(remote_temp_dir + "/mydir")
    assert fs.isfile(remote_temp_dir + "/mydir/myfile1")
    assert fs.isfile(remote_temp_dir + "/mydir/myfile2")
    fs.get(
        remote_temp_dir + "/mydir",
        str(tmp_path / "downloaded" / "mydir"),
        recursive=True,
    )
    assert (tmp_path / "downloaded" / "mydir" / "myfile1").read_text() == "myfile1"
    assert (tmp_path / "downloaded" / "mydir" / "myfile2").read_text() == "myfile2"

    assert fs.ls(remote_temp_dir, detail=False) == [
        remote_temp_dir + "/mydir",
        remote_temp_dir + "/myfile",
    ]
    assert fs.ls(posixpath.relpath(remote_temp_dir, "/data"), detail=False) == [
        remote_temp_dir + "/mydir",
        remote_temp_dir + "/myfile",
    ]

    fs.rm(remote_temp_dir + "/myfile")
    assert not fs.exists(remote_temp_dir + "/myfile")
    fs.rm(remote_temp_dir + "/mydir", recursive=True)
    assert not fs.exists(remote_temp_dir + "/mydir")
