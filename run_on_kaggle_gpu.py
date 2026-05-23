#!/usr/bin/env python3
"""
Helper script to run training on Kaggle with GPU.

Usage:
    python run_on_kaggle_gpu.py --notebook notebooks/kaggle_paper_classification.ipynb

Or interact with Kaggle Kernels API directly:
    python run_on_kaggle_gpu.py --list-datasets
    python run_on_kaggle_gpu.py --create-kernel
"""

import os
import json
import argparse
from pathlib import Path
from credentials import get_kaggle_credentials

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    raise ImportError("kaggle library not found. Install with: pip install kaggle")


def get_kaggle_api():
    """Initialize Kaggle API with credentials."""
    username, key = get_kaggle_credentials()
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    
    api = KaggleApi()
    api.authenticate()
    return api


def list_datasets(api):
    """List available datasets."""
    print("\n📊 Your Kaggle Datasets:")
    datasets = api.dataset_list(mine=True, max_size=10)
    for ds in datasets:
        print(f"  • {ds.ref} — {ds.title}")


def upload_notebook(notebook_path, kernel_name):
    """
    Upload notebook to Kaggle as a kernel.
    
    Args:
        notebook_path: Path to .ipynb file
        kernel_name: Name for the kernel on Kaggle (lowercase, no spaces)
    
    Note:
        You must manually enable GPU in Kaggle UI after uploading.
    """
    api = get_kaggle_api()
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    print(f"\n📤 Preparing to upload: {notebook_path}")
    print(f"   Kernel name: {kernel_name}")
    print("\n⚠️  Important: After upload, manually enable GPU in Kaggle UI:")
    print("   1. Go to your kernel on kaggle.com")
    print("   2. Click ⚙️ Settings (top right)")
    print("   3. Under 'Accelerator', select 'GPU'")
    print("   4. Click 'Save and Run'")
    
    # For actual kernel creation, use Kaggle UI or the `kaggle kernels` command:
    # kaggle kernels push -p /path/to/kernel/directory
    print("\n📍 To push this notebook as a kernel, use the CLI:")
    print(f"   kaggle kernels push -p /path/to/kernel/dir")


def create_kernel_metadata(notebook_name):
    """Create kernel metadata file for Kaggle."""
    kernel_meta = {
        "id": f"{notebook_name}",
        "title": f"{notebook_name.replace('_', ' ').title()}",
        "code_file": f"{notebook_name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": []
    }
    return kernel_meta


def main():
    parser = argparse.ArgumentParser(description="Run training on Kaggle with GPU")
    parser.add_argument("--list-datasets", action="store_true", help="List your Kaggle datasets")
    parser.add_argument("--notebook", type=str, help="Path to notebook to upload")
    parser.add_argument("--kernel-name", type=str, help="Name for kernel on Kaggle")
    parser.add_argument("--download-dataset", type=str, help="Download a dataset by reference")
    parser.add_argument("--info", action="store_true", help="Show setup info")
    
    args = parser.parse_args()
    
    if args.info:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RUNNING ON KAGGLE WITH GPU                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 FASTEST WAY (5 minutes):
1. Manually create kernel on Kaggle:
   → Go to kaggle.com/code/new
   → Upload your notebook
   → Click Settings → Enable GPU
   → Run!

📟 USING KAGGLE API (command-line):
1. Push notebook to Kaggle:
   kaggle kernels push -p ./kernels/my_kernel_dir
   
2. Enable GPU in kernel metadata (kernel-metadata.json):
   {
     "enable_gpu": true,
     "enable_internet": true
   }

3. Check kernel status:
   kaggle kernels status username/kernel-name

4. View output:
   kaggle kernels output username/kernel-name -p ./output

📚 DATASET MANAGEMENT:
1. List your datasets:
   python run_on_kaggle_gpu.py --list-datasets

2. Download dataset locally:
   python run_on_kaggle_gpu.py --download-dataset owner/dataset-name

3. Reference in notebook:
   /kaggle/input/dataset-name/

🔗 USEFUL LINKS:
   • Create kernel: https://www.kaggle.com/code/new
   • API docs: https://github.com/Kaggle/kaggle-api
   • Notebook settings: Settings (⚙️) → Accelerator → GPU
        """)
        return
    
    try:
        api = get_kaggle_api()
        print("✓ Kaggle authentication successful!")
        
        if args.list_datasets:
            list_datasets(api)
        
        if args.notebook:
            kernel_name = args.kernel_name or Path(args.notebook).stem
            upload_notebook(args.notebook, kernel_name)
        
        if args.download_dataset:
            print(f"\n📥 Downloading dataset: {args.download_dataset}")
            api.dataset_download_files(args.download_dataset, path='./data', unzip=True)
            print(f"✓ Downloaded to ./data/")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your .env file has KAGGLE_USERNAME and KAGGLE_KEY set")


if __name__ == "__main__":
    main()
