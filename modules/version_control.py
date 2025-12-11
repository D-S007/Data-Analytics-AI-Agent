import os
import json
import pandas as pd
from datetime import datetime
import hashlib

class VersionControl:
    def __init__(self, base_path="data_versions"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # Generate version ID
    def _generate_version_id(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save DataFrame + metadata
    def save_version(self, df: pd.DataFrame, message=""):
        version_id = self._generate_version_id()
        version_dir = os.path.join(self.base_path, version_id)
        os.makedirs(version_dir, exist_ok=True)

        # Save CSV
        file_path = os.path.join(version_dir, "data.csv")
        df.to_csv(file_path, index=False)

        # Metadata
        metadata = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "shape": df.shape,
            "message": message,
            "hash": self._hash_file(file_path)
        }

        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return version_id

    # Load a version
    def load_version(self, version_id):
        file_path = os.path.join(self.base_path, version_id, "data.csv")
        return pd.read_csv(file_path)

    # Get all versions
    def list_versions(self):
        versions = []
        for vid in sorted(os.listdir(self.base_path)):
            meta_file = os.path.join(self.base_path, vid, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    versions.append(json.load(f))
        return versions

    # File hash (verification)
    def _hash_file(self, path):
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
