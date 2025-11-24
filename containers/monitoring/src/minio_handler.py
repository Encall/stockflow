import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error


def _env_get(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


class MinioClient:
    """Lightweight S3-compatible helper for downloading and uploading single objects."""

    def __init__(self) -> None:
        load_dotenv()

        raw_endpoint = _env_get("AWS_S3_ENDPOINT_URL", "MINIO_ENDPOINT")
        access_key = _env_get("AWS_ACCESS_KEY_ID", "MINIO_ACCESS_KEY")
        secret_key = _env_get("AWS_SECRET_ACCESS_KEY", "MINIO_SECRET_KEY")
        region = _env_get("AWS_REGION", "MINIO_REGION")
        secure_flag = _env_get("AWS_S3_SECURE", "MINIO_SECURE")
        bucket_default = "stockflow"
        self.bucket_name = _env_get("AWS_S3_BUCKET", "MINIO_BUCKET") or bucket_default

        if not raw_endpoint or not access_key or not secret_key:
            raise ValueError(
                "Storage credentials missing. "
                "Set AWS_S3_ENDPOINT_URL/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (or legacy MINIO_* vars)."
            )

        endpoint, secure = self._parse_endpoint(raw_endpoint, secure_flag)

        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            secure=secure,
        )

        self._verify_bucket()

    @staticmethod
    def _parse_endpoint(endpoint: str, secure_flag: str | None) -> tuple[str, bool]:
        secure: bool | None = None
        parsed = urlparse(endpoint) if "://" in endpoint else None

        if parsed:
            endpoint_host = parsed.netloc
            secure = parsed.scheme == "https"
        else:
            endpoint_host = endpoint

        if secure_flag is not None:
            secure = secure_flag.lower() == "true"

        return endpoint_host, secure if secure is not None else True

    def _verify_bucket(self) -> None:
        """Ensure the configured bucket exists before attempting downloads."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                raise RuntimeError(f"Bucket '{self.bucket_name}' not found in object storage.")
        except S3Error as exc:
            raise RuntimeError(f"Error checking bucket '{self.bucket_name}': {exc}") from exc

    def download(self, object_name: str, target_dir: str | Path) -> Path:
        """Download a single object to the target directory and return the local path."""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / Path(object_name).name

        try:
            self.client.fget_object(self.bucket_name, object_name, local_path.as_posix())
        except S3Error as exc:
            raise RuntimeError(f"Error downloading '{object_name}' from bucket '{self.bucket_name}': {exc}") from exc

        return local_path

    def upload(self, local_path: Path | str, object_name: str) -> None:
        """Upload a local file to the configured MinIO bucket."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found for upload: {local_path}")

        try:
            self.client.fput_object(self.bucket_name, object_name, local_path.as_posix())
        except S3Error as exc:
            raise RuntimeError(f"Error uploading '{local_path}' to '{object_name}': {exc}") from exc
