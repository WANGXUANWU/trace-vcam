"""Subject-level data contracts shared by Python and R benchmark adapters."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


def _as_string_vector(values: ArrayLike, *, name: str) -> StringArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # Stable string identifiers avoid R/Python disagreements about whether a
    # numeric-looking subject ID is an integer, a float, or a factor.
    result = np.asarray([str(value) for value in array.tolist()], dtype=str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} must not contain empty identifiers")
    return result


def _update_array_hash(digest: "hashlib._Hash", name: str, array: NDArray) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    if array.dtype.kind in "USO":
        digest.update(
            json.dumps(array.astype(str).tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    else:
        canonical = np.asarray(array, dtype="<f8", order="C")
        digest.update(canonical.tobytes(order="C"))


@dataclass(frozen=True)
class SubjectDataset:
    """One-row-per-observation population VCAM data.

    ``subject_id`` is the unit of every outer split.  It is intentionally not
    inferred from row order or cluster size.  ``noise_free_target`` is optional
    because it exists in simulations but not in the MACS application.
    """

    time: FloatArray
    covariates: FloatArray
    response: FloatArray
    subject_id: StringArray
    row_id: StringArray | None = None
    noise_free_target: FloatArray | None = None
    covariate_names: tuple[str, ...] | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        covariates = np.asarray(self.covariates, dtype=float)
        response = np.asarray(self.response, dtype=float)
        subject_id = _as_string_vector(self.subject_id, name="subject_id")
        if time.ndim != 1 or response.ndim != 1:
            raise ValueError("time and response must be one-dimensional")
        if covariates.ndim != 2:
            raise ValueError("covariates must be a two-dimensional matrix")
        n_rows = len(response)
        if not (len(time) == len(subject_id) == covariates.shape[0] == n_rows):
            raise ValueError("all observation-level arrays must have the same row count")
        if n_rows == 0:
            raise ValueError("a benchmark dataset must contain at least one row")
        if not (
            np.all(np.isfinite(time))
            and np.all(np.isfinite(covariates))
            and np.all(np.isfinite(response))
        ):
            raise ValueError("time, covariates, and response must be finite")

        if self.row_id is None:
            row_id = np.asarray([f"row-{index}" for index in range(n_rows)], dtype=str)
        else:
            row_id = _as_string_vector(self.row_id, name="row_id")
        if len(row_id) != n_rows or len(np.unique(row_id)) != n_rows:
            raise ValueError("row_id must contain one unique value per observation")

        target = None
        if self.noise_free_target is not None:
            target = np.asarray(self.noise_free_target, dtype=float)
            if target.shape != (n_rows,) or not np.all(np.isfinite(target)):
                raise ValueError("noise_free_target must be a finite vector of row length")

        names = self.covariate_names
        if names is None:
            names = tuple(f"x_{index + 1}" for index in range(covariates.shape[1]))
        else:
            names = tuple(str(name) for name in names)
            if len(names) != covariates.shape[1] or len(set(names)) != len(names):
                raise ValueError("covariate_names must be unique and match the matrix width")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "noise_free_target", target)
        object.__setattr__(self, "covariate_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def subjects(self) -> StringArray:
        return np.unique(self.subject_id)

    @property
    def n_subjects(self) -> int:
        return int(len(self.subjects))

    @property
    def n_rows(self) -> int:
        return int(len(self.response))

    @property
    def data_hash(self) -> str:
        digest = hashlib.sha256()
        for name, array in (
            ("row_id", self.row_id),
            ("subject_id", self.subject_id),
            ("time", self.time),
            ("covariates", self.covariates),
            ("response", self.response),
        ):
            _update_array_hash(digest, name, np.asarray(array))
        if self.noise_free_target is not None:
            _update_array_hash(digest, "noise_free_target", self.noise_free_target)
        digest.update(json.dumps(self.covariate_names).encode("utf-8"))
        return digest.hexdigest()

    def subset_subjects(self, subjects: Iterable[str]) -> "SubjectDataset":
        requested = {str(subject) for subject in subjects}
        mask = np.asarray([subject in requested for subject in self.subject_id], dtype=bool)
        if not np.any(mask):
            raise ValueError("the subject subset is empty")
        target = None if self.noise_free_target is None else self.noise_free_target[mask]
        return SubjectDataset(
            time=self.time[mask],
            covariates=self.covariates[mask],
            response=self.response[mask],
            subject_id=self.subject_id[mask],
            row_id=self.row_id[mask],
            noise_free_target=target,
            covariate_names=self.covariate_names,
            metadata=self.metadata,
        )


def _subject_hash(subjects: Sequence[str]) -> str:
    payload = json.dumps(sorted(str(subject) for subject in subjects), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SubjectSplit:
    repeat: int
    fold: int
    seed: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        train = tuple(str(item) for item in self.train_subjects)
        test = tuple(str(item) for item in self.test_subjects)
        if not train or not test:
            raise ValueError("both train and test subject sets must be nonempty")
        if len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError("a subject may occur only once within a split side")
        if set(train).intersection(test):
            raise ValueError("train and test subjects overlap")
        object.__setattr__(self, "train_subjects", train)
        object.__setattr__(self, "test_subjects", test)

    @property
    def train_hash(self) -> str:
        return _subject_hash(self.train_subjects)

    @property
    def test_hash(self) -> str:
        return _subject_hash(self.test_subjects)

    def validate_against(self, dataset: SubjectDataset) -> None:
        observed = set(dataset.subjects.tolist())
        assigned = set(self.train_subjects).union(self.test_subjects)
        if assigned != observed:
            missing = sorted(observed - assigned)
            extra = sorted(assigned - observed)
            raise ValueError(f"split does not partition dataset subjects; missing={missing}, extra={extra}")


def make_repeated_subject_folds(
    dataset: SubjectDataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int,
) -> tuple[SubjectSplit, ...]:
    """Create deterministic folds without ever splitting a subject's rows."""

    subjects = np.asarray(sorted(dataset.subjects.tolist()), dtype=str)
    if not 2 <= n_splits <= len(subjects):
        raise ValueError("n_splits must be between 2 and the number of subjects")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    root = np.random.SeedSequence(int(seed))
    children = root.spawn(n_repeats)
    splits: list[SubjectSplit] = []
    for repeat, child in enumerate(children):
        permutation = np.random.default_rng(child).permutation(subjects)
        test_folds = np.array_split(permutation, n_splits)
        for fold, test in enumerate(test_folds):
            test_tuple = tuple(str(item) for item in test.tolist())
            test_set = set(test_tuple)
            train_tuple = tuple(str(item) for item in permutation.tolist() if str(item) not in test_set)
            split = SubjectSplit(
                repeat=repeat,
                fold=fold,
                seed=int(seed),
                train_subjects=train_tuple,
                test_subjects=test_tuple,
            )
            split.validate_against(dataset)
            splits.append(split)
    return tuple(splits)


@dataclass(frozen=True)
class ExchangeBundle:
    directory: Path
    observations_csv: Path
    split_csv: Path
    metadata_json: Path
    data_hash: str
    observations_sha256: str
    split_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exchange_bundle(
    directory: str | Path,
    dataset: SubjectDataset,
    split: SubjectSplit,
) -> ExchangeBundle:
    """Write a language-neutral subject-ID exchange bundle.

    CSV is used instead of an R-specific serialization.  The metadata contains
    content hashes, so an adapter cannot silently reorder or change the input.
    """

    split.validate_against(dataset)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    observations = output / "observations.csv"
    split_path = output / "subject_split.csv"
    metadata_path = output / "metadata.json"

    header = ["row_id", "subject_id", "time", "response"]
    if dataset.noise_free_target is not None:
        header.append("noise_free_target")
    header.extend(dataset.covariate_names)
    with observations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for index in range(dataset.n_rows):
            row: list[object] = [
                dataset.row_id[index],
                dataset.subject_id[index],
                format(float(dataset.time[index]), ".17g"),
                format(float(dataset.response[index]), ".17g"),
            ]
            if dataset.noise_free_target is not None:
                row.append(format(float(dataset.noise_free_target[index]), ".17g"))
            row.extend(format(float(value), ".17g") for value in dataset.covariates[index])
            writer.writerow(row)

    partitions = {subject: "train" for subject in split.train_subjects}
    partitions.update({subject: "test" for subject in split.test_subjects})
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject_id", "partition", "repeat", "fold", "seed"])
        for subject in sorted(partitions):
            writer.writerow([subject, partitions[subject], split.repeat, split.fold, split.seed])

    observations_hash = _file_sha256(observations)
    split_hash = _file_sha256(split_path)
    payload = {
        "schema_version": "vcam-subject-exchange/1",
        "data_hash": dataset.data_hash,
        "n_rows": dataset.n_rows,
        "n_subjects": dataset.n_subjects,
        "covariate_names": list(dataset.covariate_names),
        "train_subject_hash": split.train_hash,
        "test_subject_hash": split.test_hash,
        "observations_sha256": observations_hash,
        "subject_split_sha256": split_hash,
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return ExchangeBundle(
        directory=output,
        observations_csv=observations,
        split_csv=split_path,
        metadata_json=metadata_path,
        data_hash=dataset.data_hash,
        observations_sha256=observations_hash,
        split_sha256=split_hash,
    )


def read_exchange_bundle(directory: str | Path) -> tuple[SubjectDataset, SubjectSplit]:
    """Read and verify a bundle before an adapter consumes it."""

    source = Path(directory)
    observations = source / "observations.csv"
    split_path = source / "subject_split.csv"
    metadata_path = source / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if _file_sha256(observations) != metadata["observations_sha256"]:
        raise ValueError("observations.csv hash does not match metadata")
    if _file_sha256(split_path) != metadata["subject_split_sha256"]:
        raise ValueError("subject_split.csv hash does not match metadata")

    with observations.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reserved = {"row_id", "subject_id", "time", "response", "noise_free_target"}
    covariate_names = tuple(name for name in rows[0] if name not in reserved)
    has_target = "noise_free_target" in rows[0]
    dataset = SubjectDataset(
        row_id=np.asarray([row["row_id"] for row in rows], dtype=str),
        subject_id=np.asarray([row["subject_id"] for row in rows], dtype=str),
        time=np.asarray([float(row["time"]) for row in rows]),
        response=np.asarray([float(row["response"]) for row in rows]),
        covariates=np.asarray(
            [[float(row[name]) for name in covariate_names] for row in rows], dtype=float
        ),
        noise_free_target=(
            np.asarray([float(row["noise_free_target"]) for row in rows])
            if has_target
            else None
        ),
        covariate_names=covariate_names,
    )
    if dataset.data_hash != metadata["data_hash"]:
        raise ValueError("reconstructed dataset hash does not match metadata")

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    train = tuple(row["subject_id"] for row in split_rows if row["partition"] == "train")
    test = tuple(row["subject_id"] for row in split_rows if row["partition"] == "test")
    first = split_rows[0]
    split = SubjectSplit(
        repeat=int(first["repeat"]),
        fold=int(first["fold"]),
        seed=int(first["seed"]),
        train_subjects=train,
        test_subjects=test,
    )
    split.validate_against(dataset)
    if split.train_hash != metadata["train_subject_hash"] or split.test_hash != metadata["test_subject_hash"]:
        raise ValueError("reconstructed split hashes do not match metadata")
    return dataset, split
