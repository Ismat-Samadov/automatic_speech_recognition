"""
Download Azerbaijani ASR Dataset from Hugging Face

This script downloads the dataset with retry logic to handle 503 errors.
Run this script before the training notebook.

Usage:
    python download_data.py
"""

import os
import ssl
import sys
import time

# ============================================================
# SSL and Environment Configuration
# ============================================================
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Patch requests
import requests
_orig_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

# ============================================================
# Configuration
# ============================================================
DATASET_NAME = "LocalDoc/azerbaijani_asr"
LOCAL_DATA_DIR = "./data"
MAX_RETRIES = 10
RETRY_DELAY = 30  # seconds

# ============================================================
# Download with retry logic
# ============================================================
def download_dataset():
    from datasets import load_dataset

    print(f"Downloading dataset: {DATASET_NAME}")
    print(f"This may take a while due to server issues...")
    print(f"Will retry up to {MAX_RETRIES} times with {RETRY_DELAY}s delay")
    print("=" * 50)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\nAttempt {attempt}/{MAX_RETRIES}...")

            # Try streaming first to check connectivity
            dataset = load_dataset(
                DATASET_NAME,
                streaming=False,
                trust_remote_code=True,
            )

            print(f"\nDataset downloaded successfully!")
            print(f"Dataset structure: {dataset}")

            # Save to disk
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
            dataset.save_to_disk(LOCAL_DATA_DIR)
            print(f"\nDataset saved to: {LOCAL_DATA_DIR}")

            return True

        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg[:200]}...")

            if "503" in error_msg or "Service Unavailable" in error_msg:
                if attempt < MAX_RETRIES:
                    print(f"Server unavailable. Waiting {RETRY_DELAY}s before retry...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("\nMax retries reached. Server may be down.")
                    print("Try again later or download manually.")
                    return False
            else:
                print(f"Unexpected error: {e}")
                return False

    return False


def download_streaming():
    """Alternative: Download using streaming mode (processes data on-the-fly)"""
    from datasets import load_dataset

    print(f"Loading dataset in streaming mode: {DATASET_NAME}")
    print("This downloads data as needed during training...")

    try:
        dataset = load_dataset(DATASET_NAME, streaming=True)
        print(f"Streaming dataset ready!")
        print(f"Train split available: {'train' in dataset}")

        # Save config for notebook
        config = {
            "dataset_name": DATASET_NAME,
            "streaming": True,
        }

        import json
        with open("dataset_config.json", "w") as f:
            json.dump(config, f)

        print("Config saved to dataset_config.json")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Azerbaijani ASR Dataset Downloader")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "--streaming":
        success = download_streaming()
    else:
        success = download_dataset()

    if success:
        print("\n" + "=" * 50)
        print("SUCCESS! You can now run the training notebook.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("FAILED. Options:")
        print("1. Try again later (HF servers may be overloaded)")
        print("2. Run with --streaming flag: python download_data.py --streaming")
        print("3. Download dataset manually from HuggingFace website")
        print("=" * 50)
        sys.exit(1)
