
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download


def download_ds(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id="visual-layer/imagenet-1k-vl-enriched",
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print("Downloaded to:", snapshot_path)


def main(out_dir: Path):
    download_ds(out_dir)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Download the imagenet-1k-vl-enriched dataset snapshot to a local directory"
    )
    p.add_argument(
        "--out_dir",
        "-o",
        type=Path,
        required=True,
        help="Local directory where the dataset snapshot will be downloaded (default: ./data)",
    )
    return p.parse_args()

# uv run -m gradientlab.experiments.exp20251025_0_vlm_20m_in1k.dataprep_download
if __name__ == "__main__":
    args = _parse_args()
    main(args.out_dir)