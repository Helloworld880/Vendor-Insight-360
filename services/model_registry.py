from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config.settings import get_settings


settings = get_settings()


class ModelRegistryError(Exception):
    pass


@dataclass
class LoadedModel:
    model_name: str
    version: str
    model: RandomForestClassifier
    metadata: dict


class ModelRegistry:
    def __init__(self, registry_root: Path | None = None) -> None:
        self.registry_root = registry_root or settings.model_registry_path
        self.registry_root.mkdir(parents=True, exist_ok=True)

    def _model_root(self, model_name: str) -> Path:
        path = self.registry_root / model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _version_root(self, model_name: str, version: str) -> Path:
        return self._model_root(model_name) / version

    def train_and_register(self, model_name: str, dataset_path: Path | None = None) -> dict:
        source_path = dataset_path or settings.training_dataset_path
        frame = pd.read_csv(source_path)
        required_columns = [
            "delivery_rate",
            "quality_score",
            "cost_efficiency",
            "on_time_rate",
            "cost_variance",
            "reliability",
            "performance_score",
            "risk_label",
        ]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ModelRegistryError(f"Training dataset is missing required columns: {missing}")

        feature_columns = required_columns[:-1]
        x_train, x_test, y_train, y_test = train_test_split(
            frame[feature_columns],
            frame["risk_label"],
            test_size=0.25,
            random_state=42,
            stratify=frame["risk_label"],
        )

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=2,
            random_state=42,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        trained_at = datetime.now(timezone.utc)
        version = trained_at.strftime("v%Y%m%d%H%M%S")
        version_root = self._version_root(model_name, version)
        version_root.mkdir(parents=True, exist_ok=False)

        model_path = version_root / "model.joblib"
        metadata_path = version_root / "metadata.json"
        joblib.dump(model, model_path)
        metadata = {
            "model_name": model_name,
            "version": version,
            "features": feature_columns,
            "accuracy": round(float(accuracy), 6),
            "training_timestamp": trained_at.isoformat(),
            "record_count": int(len(frame)),
            "target_column": "risk_label",
            "model_path": str(model_path.as_posix()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def list_versions(self, model_name: str) -> list[dict]:
        model_root = self._model_root(model_name)
        versions: list[dict] = []
        for version_dir in sorted([item for item in model_root.iterdir() if item.is_dir()], reverse=True):
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                versions.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        return versions

    def load_model(self, model_name: str, version: str) -> LoadedModel:
        version_root = self._version_root(model_name, version)
        model_path = version_root / "model.joblib"
        metadata_path = version_root / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise ModelRegistryError(f"Model version '{version}' not found for '{model_name}'.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        return LoadedModel(model_name=model_name, version=version, model=model, metadata=metadata)

    def load_latest_model(self, model_name: str) -> LoadedModel:
        versions = self.list_versions(model_name)
        if not versions:
            raise ModelRegistryError(f"No versions registered for '{model_name}'.")
        return self.load_model(model_name, versions[0]["version"])

    def ensure_model(self, model_name: str) -> dict:
        versions = self.list_versions(model_name)
        if versions:
            return versions[0]
        return self.train_and_register(model_name=model_name)
