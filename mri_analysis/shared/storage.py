from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol
from urllib.parse import urlparse

import boto3


class StorageError(RuntimeError):
    """Raised when the storage backend fails."""


class StorageClient(Protocol):
    bucket: str

    def ensure_bucket(self) -> None:
        ...

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        ...

    def upload_file(self, local_path: Path, key: str, content_type: Optional[str] = None) -> str:
        ...

    def download_file(self, uri: str, local_path: Path) -> None:
        ...

    def read_bytes(self, uri: str) -> bytes:
        ...


def guess_mime_type(filename: str, default: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or default


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise StorageError(f"Unsupported storage URI scheme: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise StorageError(f"Invalid storage URI: {uri}")
    return bucket, key


@dataclass
class S3StorageClient:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    region: str = "us-east-1"

    def __post_init__(self) -> None:
        self._resource = boto3.resource(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            self._client.create_bucket(Bucket=self.bucket)

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType=content_type)
        return f"s3://{self.bucket}/{key}"

    def upload_file(self, local_path: Path, key: str, content_type: Optional[str] = None) -> str:
        extra: Dict[str, str] = {}
        if content_type:
            extra["ContentType"] = content_type
        if extra:
            self._client.upload_file(str(local_path), self.bucket, key, ExtraArgs=extra)
        else:
            self._client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_file(self, uri: str, local_path: Path) -> None:
        bucket, key = parse_s3_uri(uri)
        if bucket != self.bucket:
            raise StorageError(f"Unexpected bucket {bucket}, expected {self.bucket}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, str(local_path))

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = parse_s3_uri(uri)
        if bucket != self.bucket:
            raise StorageError(f"Unexpected bucket {bucket}, expected {self.bucket}")
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()


@dataclass
class LocalStorageClient:
    root: Path
    bucket: str = "mri-analysis"

    def ensure_bucket(self) -> None:
        self._bucket_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _bucket_dir(self) -> Path:
        return self.root / self.bucket

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> str:
        del content_type
        path = self._bucket_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return f"s3://{self.bucket}/{key}"

    def upload_file(self, local_path: Path, key: str, content_type: Optional[str] = None) -> str:
        del content_type
        path = self._bucket_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(local_path.read_bytes())
        return f"s3://{self.bucket}/{key}"

    def download_file(self, uri: str, local_path: Path) -> None:
        bucket, key = parse_s3_uri(uri)
        if bucket != self.bucket:
            raise StorageError(f"Unexpected bucket {bucket}, expected {self.bucket}")
        source = self._bucket_dir / key
        if not source.exists():
            raise StorageError(f"Object does not exist: {uri}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(source.read_bytes())

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = parse_s3_uri(uri)
        if bucket != self.bucket:
            raise StorageError(f"Unexpected bucket {bucket}, expected {self.bucket}")
        source = self._bucket_dir / key
        if not source.exists():
            raise StorageError(f"Object does not exist: {uri}")
        return source.read_bytes()


def build_storage_from_env(prefix: str = "STORAGE_") -> StorageClient:
    mode = os.getenv(f"{prefix}MODE", "local").lower()
    bucket = os.getenv(f"{prefix}BUCKET", "mri-analysis")
    if mode == "local":
        root = Path(os.getenv(f"{prefix}ROOT", ".data/storage"))
        client = LocalStorageClient(root=root, bucket=bucket)
    else:
        client = S3StorageClient(
            endpoint_url=os.getenv(f"{prefix}ENDPOINT_URL", "http://minio:9000"),
            access_key=os.getenv(f"{prefix}ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv(f"{prefix}SECRET_KEY", "minioadmin"),
            bucket=bucket,
            region=os.getenv(f"{prefix}REGION", "us-east-1"),
        )
    client.ensure_bucket()
    return client
