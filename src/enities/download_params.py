from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    paths: list[str]
    output_folder: str
    s3_bucket: str = field(default="for-dvc")