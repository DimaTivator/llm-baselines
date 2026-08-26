import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

if importlib.util.find_spec("tiktoken") is None:
    tiktoken = ModuleType("tiktoken")
    tiktoken.get_encoding = lambda _: object()
    sys.modules["tiktoken"] = tiktoken

if (
    importlib.util.find_spec("datasets") is None
    or importlib.util.find_spec("datasets").loader is None
):
    datasets = ModuleType("datasets")
    datasets.load_dataset = None
    sys.modules["datasets"] = datasets

from data.fineweb_edu import get_fineweb_edu_data


def test_require_tokenized_returns_complete_dataset(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    train_path = tokenized_dir / "train.bin"
    val_path = tokenized_dir / "val.bin"
    train_path.write_bytes(b"train")
    val_path.write_bytes(b"val")

    paths = get_fineweb_edu_data(
        str(tmp_path / "parquets"),
        tokenized_data_dir=str(tokenized_dir),
        require_tokenized=True,
    )

    assert paths == {"train": str(train_path), "val": str(val_path)}


@pytest.mark.parametrize("missing_name", ["train.bin", "val.bin"])
def test_require_tokenized_rejects_missing_split(
    tmp_path: Path, missing_name: str
) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    for name in ("train.bin", "val.bin"):
        if name != missing_name:
            (tokenized_dir / name).write_bytes(b"tokens")

    with pytest.raises(FileNotFoundError, match=missing_name):
        get_fineweb_edu_data(
            str(tmp_path / "parquets"),
            tokenized_data_dir=str(tokenized_dir),
            require_tokenized=True,
        )


def test_require_tokenized_rejects_empty_split(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    (tokenized_dir / "train.bin").write_bytes(b"tokens")
    (tokenized_dir / "val.bin").touch()

    with pytest.raises(FileNotFoundError, match="val.bin"):
        get_fineweb_edu_data(
            str(tmp_path / "parquets"),
            tokenized_data_dir=str(tokenized_dir),
            require_tokenized=True,
        )
