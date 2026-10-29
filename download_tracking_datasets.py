"""Download the tracking splits to D: with junctions in project/data."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request
import download_datasets as downloader

def main():
    storage = Path('D:/deeplearning-datasets')
    downloader.ARCHIVES = storage / '_downloads'
    downloader.ARCHIVES.mkdir(parents=True, exist_ok=True)
    jobs = []
    for repo, folder in [('mrhwl123/BDD100kMOT', 'BDD100K-MOT'), ('vanthanh/VisDrone2019-MOT', 'VisDrone2019-MOT')]:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/' + repo + '/tree/main', timeout=60) as response:
            listing = json.load(response)
        for item in listing:
            if not item['path'].endswith('.zip'):
                continue
            jobs.append((item['path'], 'https://huggingface.co/datasets/' + repo + '/resolve/main/' + item['path'] + '?download=true', str(storage / folder), item['size'], 'sha256', item['lfs']['oid']))
    (downloader.ROOT / 'tracking_download_sources.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    print('TOTAL TRACKING ARCHIVE BYTES', sum(job[3] for job in jobs), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(downloader.download, jobs))
    print('ALL TRACKING DOWNLOADS VERIFIED AND EXTRACTED', flush=True)

if __name__ == '__main__':
    main()
