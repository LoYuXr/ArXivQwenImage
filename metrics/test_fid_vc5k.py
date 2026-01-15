#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from io import BytesIO
from typing import List, Tuple

# --- Canva 环境依赖（按你的示例） ---
os.environ.setdefault('AWS_PROFILE', 'prod')
sys.path.insert(0, "/home/coder/work/canva")
from tools.build.python.lib.aws.src.aws import SessionFactory  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.fs as pafs  # noqa: E402
from PIL import Image  # noqa: E402

# pytorch-fid 的 Python API
from pytorch_fid.fid_score import calculate_fid_given_paths  # noqa: E402


DEVBOX_ENV_VAR = "CODER_AGENT_TOKEN"
TRAINING_PLATFORM_ENV_VAR = "ANYSCALE_CLOUD_ID"


def create_s3_filesystem(model_trainer_arn: str, aws_region: str) -> pafs.S3FileSystem:
    """Create an S3FileSystem instance appropriate for the current environment."""
    if os.environ.get(TRAINING_PLATFORM_ENV_VAR):
        # Running on Anyscale - use role-based authentication
        fs = pafs.S3FileSystem(
            role_arn=model_trainer_arn,
            region=aws_region,
        )
    else:
        print('[FID] running locally (session credentials).')
        # Running locally - use session credentials
        session = SessionFactory().create()
        credentials = session.get_credentials()

        if credentials is None:
            raise ValueError("No AWS credentials found in the session")

        fs = pafs.S3FileSystem(
            secret_key=credentials.secret_key,
            access_key=credentials.access_key,
            region=aws_region,
            session_token=credentials.token,
        )

    return fs


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Return (bucket, key_prefix)."""
    p = urlparse(uri)
    if p.scheme != 's3' or not p.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = p.netloc
    key = p.path.lstrip('/')
    return bucket, key


def list_images_local(folder: str) -> List[str]:
    all_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                all_paths.append(os.path.join(root, f))
    return all_paths


def list_images_s3(fs: pafs.S3FileSystem, s3_uri: str) -> List[Tuple[str, str]]:
    """Return list of (s3_uri, relative_path) to preserve structure under temp dir."""
    bucket, key_prefix = parse_s3_uri(s3_uri)
    selector = pafs.FileSelector(f"{bucket}/{key_prefix}", recursive=True)
    infos = fs.get_file_info(selector)
    out = []
    base_len = len(key_prefix.rstrip('/')) + 1 if key_prefix else 0
    for info in infos:
        if info.is_file:
            ext = os.path.splitext(info.path.lower())[1]
            if ext in IMG_EXTS:
                # info.path format: "bucket/key..."
                # convert to s3 uri
                _, full_key = info.path.split('/', 1)
                rel = full_key[base_len:] if base_len and full_key.startswith(key_prefix.rstrip('/') + '/') else os.path.basename(full_key)
                out.append((f"s3://{bucket}/{full_key}", rel))
    return out


def download_one(fs: pafs.S3FileSystem, s3_uri: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with fs.open_input_stream(s3_uri.replace("s3://", "")) as src, open(local_path, 'wb') as dst:
        # Stream copy
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def mirror_s3_to_temp(fs: pafs.S3FileSystem, s3_uri: str, tmp_root: str, max_workers: int = 16) -> str:
    files = list_images_s3(fs, s3_uri)
    if not files:
        raise ValueError(f"No images found under {s3_uri}")

    target_root = os.path.join(tmp_root, "s3_" + str(abs(hash(s3_uri)) % (10**8)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s3_p, rel in files:
            local_p = os.path.join(target_root, rel)
            futures.append(ex.submit(download_one, fs, s3_p, local_p))
        for f in as_completed(futures):
            _ = f.result()
    return target_root


def ensure_local_dir(fs: pafs.S3FileSystem, maybe_uri: str, tmp_root: str, max_workers: int) -> str:
    """Return a local on-disk directory containing images."""
    if is_s3_uri(maybe_uri):
        return mirror_s3_to_temp(fs, maybe_uri, tmp_root, max_workers)
    else:
        if not os.path.isdir(maybe_uri):
            raise ValueError(f"Local path not found or not a directory: {maybe_uri}")
        # validate images exist
        if not list_images_local(maybe_uri):
            raise ValueError(f"No images found under local path: {maybe_uri}")
        return maybe_uri


def main():
    parser = argparse.ArgumentParser(description="Compute FID for two folders (local or s3).")
    parser.add_argument("--src", required=True, help="Source images folder. Supports local path or s3://bucket/prefix")
    parser.add_argument("--dst", required=True, help="Destination images folder. Supports local path or s3://bucket/prefix")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model-trainer-arn", default="arn:aws:iam::051687089423:role/service.core-cn-model-trainer")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
                        help="cuda or cpu")
    parser.add_argument("--max-download-workers", type=int, default=32)
    parser.add_argument("--cleanup", action="store_true", help="Force delete temporary directories after run.")
    args = parser.parse_args()

    # S3 FS (even本地路径也无影响；我们只在需要时用到)
    fs = create_s3_filesystem(model_trainer_arn=args.model_trainer_arn, aws_region=args.aws_region)

    tmp_root = tempfile.mkdtemp(prefix="fid_tmp_")
    local_dirs = []
    try:
        src_local = ensure_local_dir(fs, args.src, tmp_root, args.max_download_workers)
        dst_local = ensure_local_dir(fs, args.dst, tmp_root, args.max_download_workers)
        local_dirs = [src_local, dst_local]

        # 调用 pytorch-fid 的官方 API
        fid = calculate_fid_given_paths(
            paths=local_dirs,
            batch_size=args.batch_size,
            device=args.device,
            dims=2048,
            num_workers=args.num_workers,
        )
        print(f"[FID] FID({args.src} vs {args.dst}) = {fid:.6f}")
    finally:
        # 只有当我们创建了临时镜像目录才清理（避免误删真实本地目录）
        if args.cleanup:
            for d in set(local_dirs):
                if d.startswith(tmp_root):
                    shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(tmp_root, ignore_errors=True)


# 简单本地/环境连通性测试（可选）
def unit_test():
    fs = create_s3_filesystem(
        model_trainer_arn='arn:aws:iam::051687089423:role/service.core-cn-model-trainer',
        aws_region="us-east-1"
    )
    # 示例：只列一下 S3 里的图片文件数量，验证能读
    test_uri = "s3://your-bucket/your-prefix/"
    files = list_images_s3(fs, test_uri)
    print(f"Found {len(files)} images under {test_uri}")


if __name__ == "__main__":
    main()
