"""
Downloads Shin 2017 simultaneous EEG+NIRS dataset from depositonce TU Berlin.
Run from the project root:  python data/download_dataset.py
"""

import ssl
import urllib.request
import zipfile
from pathlib import Path

# depositonce.tu-berlin.de uses a self-signed certificate chain
# SSL verification is disabled for this institutional repository only
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

BASE_URL = "https://depositonce.tu-berlin.de"

# Direct bitstream URLs extracted from depositonce page
NIRS_FILES = {
    "VP001": "/bitstreams/652218ac-7fa0-413d-b9f1-b8fdc9d82bbd/download",
    "VP002": "/bitstreams/2e255b4c-3976-483f-a48a-a060de1870f6/download",
    "VP003": "/bitstreams/18f5d8f7-48d6-4946-8503-b9c07e5e9a7e/download",
    "VP004": "/bitstreams/dfc20a5f-349a-4a6e-bf43-b660a4cf03c8/download",
    "VP005": "/bitstreams/1b95f100-adb1-49c2-a763-4b94391f9b19/download",
    "VP006": "/bitstreams/b2f98ea9-3190-4bd7-9a82-78b50c7d165f/download",
    "VP007": "/bitstreams/01321865-4938-4508-b9b8-4e8ad6e2d385/download",
    "VP008": "/bitstreams/d22a01eb-d1f3-4ea2-b6dd-e2dedf261841/download",
    "VP009": "/bitstreams/4b009d7a-d448-4f53-9859-9d3088cf9124/download",
    "VP010": "/bitstreams/1c2b4da0-6614-4b42-a735-a116d56a8600/download",
    "VP011": "/bitstreams/c9d18d7b-a998-451e-b582-c9c43848b3d3/download",
    # VP012 and VP013 NIRS missing from dataset (excluded subjects)
    "VP014": "/bitstreams/c27f68c6-abba-44f1-9970-79818da476b0/download",
    "VP015": "/bitstreams/f0156ff3-7c2a-4053-b35e-477bdd2d2e03/download",
    "VP016": "/bitstreams/c857dbdf-73ea-4e3a-bb37-f5acaf23799d/download",
    "VP017": "/bitstreams/baf3045e-6eaf-4e55-b6b6-1fd458247415/download",
    "VP018": "/bitstreams/d078f732-ee7e-4eb1-ab64-b9a5201ef75a/download",
    "VP019": "/bitstreams/68928988-fea0-4dd9-9e97-8715ac495cc9/download",
    "VP020": "/bitstreams/150ee2e6-5ccb-4092-9ba2-457ea359cdd8/download",
    "VP021": "/bitstreams/703ad308-8b44-4180-b791-9dcb36ebde92/download",
    "VP022": "/bitstreams/a3906d9d-5017-4abb-84a0-865975a0517f/download",
    "VP023": "/bitstreams/bc190edb-9e35-4a22-9a32-eadf34c33096/download",
    "VP024": "/bitstreams/e9398a21-24d0-4a2e-8f0f-74cdf89f9efd/download",
    "VP025": "/bitstreams/09a6bafe-6e57-4e4c-aec1-c9bc130859c1/download",
    "VP026": "/bitstreams/53c01634-c99b-469a-a7d0-ae24410e9387/download",
}

EEG_FILES = {
    "VP001": "/bitstreams/e9981a66-bd43-4df6-be5d-00a6b3e92d51/download",
    "VP002": "/bitstreams/0d765119-5717-4de4-ad9c-57e5ca428365/download",
    "VP003": "/bitstreams/f8d1edc0-6f85-4352-9aff-5a74c401d3b8/download",
    "VP004": "/bitstreams/4b03d387-0a2d-40d3-be43-4a202395fd3a/download",
    "VP005": "/bitstreams/ae158b8c-0937-4623-b04a-dab5673f13ac/download",
    "VP006": "/bitstreams/c8374a8a-97c4-44f1-a47a-0462ba1b3a54/download",
    "VP007": "/bitstreams/a4fbd05e-362a-42c6-b810-e7621e297671/download",
    "VP008": "/bitstreams/0236ddd3-98b8-473f-add5-1c44e88b6c80/download",
    "VP009": "/bitstreams/cbfb7549-c845-43d6-9a9a-056a1ae77ccb/download",
    "VP010": "/bitstreams/e3149970-06a5-4988-8734-fa07f6c22f5c/download",
    "VP011": "/bitstreams/3aea4ad7-6285-4f3d-9475-75adcb8d6695/download",
    # VP012/VP013 EEG exist on the server but are excluded since they have no NIRS counterpart
    "VP014": "/bitstreams/b32f32a6-eb99-43e0-8f50-c1bd387a9c94/download",
    "VP015": "/bitstreams/a1d8ff9c-a7f7-4a97-b4e6-ef96e27e83e3/download",
    "VP016": "/bitstreams/88e9de88-5b6e-4c93-8788-707826f86cf0/download",
    "VP017": "/bitstreams/74060e0e-013b-4444-b36f-d66392a30ad8/download",
    "VP018": "/bitstreams/5f65bb8f-3fd1-4ea5-886d-4bf034e39df3/download",
    "VP019": "/bitstreams/e40c262b-38e0-41a3-8a3e-3cdc57c09344/download",
    "VP020": "/bitstreams/8d95f44c-f42b-462f-b306-0c4dccc37814/download",
    "VP021": "/bitstreams/7145bcdc-52c8-49fb-82aa-d1a10bf47f68/download",
    "VP022": "/bitstreams/7bf7b313-02b3-4030-98f5-29179810e6c6/download",
    "VP023": "/bitstreams/f36cc31e-40d5-4e98-9795-22d83b29f8ca/download",
    "VP024": "/bitstreams/161864e8-8dcb-4852-99d4-37a91b853e90/download",
    "VP025": "/bitstreams/c18e90ce-0942-463f-a8f0-e2d679c1dd14/download",
    # VP026 EEG bitstream not found in current repository listing — NIRS only
}

RAW_DIR = Path(__file__).parent / "raw"


def download_file(url, dest_path, label):
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}, skipping")
        return True
    print(f"  Downloading {label}...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as response:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def unzip_to(zip_path, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()  # remove zip after extraction


def download_subject(subj, modality, url_map):
    url = BASE_URL + url_map[subj]
    subj_dir = RAW_DIR / subj / modality
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    existing = list(subj_dir.glob("*"))
    if existing:
        print(f"  {subj} {modality}: already extracted ({len(existing)} files), skipping")
        return True

    zip_path = RAW_DIR / f"{subj}-{modality}.zip"
    ok = download_file(url, zip_path, f"{subj}-{modality}.zip")
    if ok:
        print(f"  Extracting {subj}-{modality}.zip...", end=" ", flush=True)
        try:
            unzip_to(zip_path, subj_dir)
            print("done")
        except Exception as e:
            print(f"FAILED to unzip: {e}")
            return False
    return ok


if __name__ == "__main__":
    import sys

    # Allow partial download: python download_dataset.py VP001 VP002 VP003
    subjects_to_download = sys.argv[1:] if len(sys.argv) > 1 else sorted(NIRS_FILES.keys())

    print(f"Downloading {len(subjects_to_download)} subjects to {RAW_DIR}")
    print(f"Note: VP012 and VP013 have no NIRS data (excluded from dataset)\n")

    failed = []
    for subj in subjects_to_download:
        print(f"\n{subj}:")
        if subj in NIRS_FILES:
            ok = download_subject(subj, "NIRS", NIRS_FILES)
            if not ok:
                failed.append(f"{subj}-NIRS")
        if subj in EEG_FILES:
            ok = download_subject(subj, "EEG", EEG_FILES)
            if not ok:
                failed.append(f"{subj}-EEG")
        else:
            print(f"  {subj} EEG: URL not in script yet, download manually")

    print("\n" + "="*50)
    if failed:
        print(f"FAILED downloads: {failed}")
        print("Re-run the script to retry failed downloads.")
    else:
        print("All downloads complete.")
    print(f"Data saved to: {RAW_DIR}")
