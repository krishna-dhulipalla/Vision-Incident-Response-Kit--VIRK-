from pathlib import Path

import pytest

from virk.repro.storage import LocalStorage


def test_local_storage_accepts_bundle_already_in_output_dir(tmp_path):
    bundle = tmp_path / "incident_123.zip"
    bundle.write_bytes(b"bundle contents")
    uri = LocalStorage(str(tmp_path)).save(bundle, bundle.name)
    assert Path(uri) == bundle.absolute()
    assert bundle.read_bytes() == b"bundle contents"


def test_local_storage_copies_from_other_directory(tmp_path):
    bundle = tmp_path / "source.zip"
    bundle.write_bytes(b"bundle contents")
    uri = LocalStorage(str(tmp_path / "destination")).save(bundle, "incident.zip")
    assert Path(uri).read_bytes() == b"bundle contents"


def test_local_storage_missing_same_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalStorage(str(tmp_path)).save(tmp_path / "missing.zip", "missing.zip")
