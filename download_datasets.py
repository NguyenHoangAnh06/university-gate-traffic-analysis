"""Download classroom detection datasets beside the existing MOT20 directory."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent / 'data'
ARCHIVES = ROOT / '_archives' / 'detection'
ARCHIVES.mkdir(parents=True, exist_ok=True)
EXTRACTION_SLOTS = threading.Semaphore(2)

def fetch_text(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())

def ranged_download(url, partial, size):
    """Resume independent HTTP ranges, validating each returned byte range."""
    chunk_size = 16 * 1024 * 1024
    ledger = partial.with_suffix('.ranges.json')
    completed = set(json.loads(ledger.read_text())) if ledger.exists() else set()
    with partial.open('r+b' if partial.exists() else 'w+b') as output:
        output.truncate(size)
        pending = [i for i in range((size + chunk_size - 1) // chunk_size) if i not in completed]
        def get_chunk(index):
            start = index * chunk_size
            end = min(size, start + chunk_size) - 1
            for attempt in range(5):
                try:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req, timeout=45) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server returned an unexpected range')
                        data = response.read()
                    if len(data) != end - start + 1:
                        raise ValueError('Incomplete range')
                    return index, start, data
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            # Bound queued results to avoid retaining a whole archive in memory.
            futures = {pool.submit(get_chunk, i) for i in pending[:16]}
            remaining = iter(pending[16:])
            while futures:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    index, start, data = future.result()
                    output.seek(start)
                    output.write(data)
                    output.flush()
                    completed.add(index)
                    temporary_ledger = ledger.with_suffix('.json.tmp')
                    temporary_ledger.write_text(json.dumps(sorted(completed)))
                    temporary_ledger.replace(ledger)
                    if len(completed) % 8 == 0 or len(completed) == (size + chunk_size - 1) // chunk_size:
                        print('PROGRESS', partial.name, round(min(len(completed)*chunk_size,size)/size*100,1), '%', flush=True)
                    next_index = next(remaining, None)
                    if next_index is not None:
                        futures.add(pool.submit(get_chunk, next_index))

def download(job):
    name, url, folder, expected_size, algorithm, checksum = job
    archive = ARCHIVES / name
    partial = archive.with_suffix('.zip.part')
    destination = ROOT / folder
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / (name + '.verified.json')
    if marker.exists():
        print('ALREADY VERIFIED', name, flush=True)
        return
    for attempt in range(5):
        try:
            if not archive.exists():
                if 'huggingface.co/' in url:
                    print('RESUMING RANGED DOWNLOAD', name, flush=True)
                    ranged_download(url, partial, expected_size)
                    partial.rename(archive)
            if not archive.exists():
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url + ('&' if '?' in url else '?') + 'download_attempt=' + str(time.time_ns()), headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset and response.status != 206:
                        offset = 0
                    total = offset + int(response.headers.get('Content-Length', 0))
                    print('DOWNLOADING', name, 'bytes', total, 'resume', offset, flush=True)
                    with partial.open('ab' if offset else 'wb') as output:
                        while chunk := response.read(4 * 1024 * 1024):
                            output.write(chunk)
                if expected_size and partial.stat().st_size != expected_size:
                    raise ValueError(f'{name}: unexpected size {partial.stat().st_size}')
                partial.rename(archive)
            if checksum:
                digest = hashlib.new(algorithm)
                with archive.open('rb') as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != checksum:
                    raise ValueError(f'{name}: checksum mismatch')
            print('VERIFYING ZIP', name, flush=True)
            with EXTRACTION_SLOTS, zipfile.ZipFile(archive) as zipped:
                files = zipped.infolist()
                for info in files:
                    target = (destination / info.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError('Unsafe ZIP path: ' + info.filename)
                print('EXTRACTING', name, 'entries', len(files), 'uncompressed', sum(x.file_size for x in files), flush=True)
                # Precreate parents so independent extraction threads cannot race
                # while making the same image directory. ZipFile checks each CRC.
                parents = {(destination / info.filename).parent for info in files}
                for parent in parents:
                    parent.mkdir(parents=True, exist_ok=True)
                regular_files = [info for info in files if not info.is_dir()]
                for info in files:
                    if info.is_dir():
                        (destination / info.filename).mkdir(parents=True, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as extractors:
                    for count, _ in enumerate(extractors.map(lambda info: zipped.extract(info, destination), regular_files), 1):
                        if count % 10000 == 0:
                            print('EXTRACTED', name, count, '/', len(regular_files), flush=True)
            marker.write_text(json.dumps({'archive': name, 'source': url, 'size': archive.stat().st_size, 'checksum': checksum, 'algorithm': algorithm, 'entries': len(files), 'uncompressed_bytes': sum(x.file_size for x in files)}, indent=2), encoding='utf-8')
            print('COMPLETE', name, flush=True)
            return
        except Exception as error:
            print('RETRY', name, attempt + 1, repr(error), flush=True)
            if attempt == 4:
                raise
            time.sleep(3)

def main():
    jobs = []
    bdd_repo = 'Myszz/BDD100k'
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{bdd_repo}/tree/main', timeout=60) as response:
        bdd_files = json.load(response)
    for item in bdd_files:
        if item['path'] in ('train.zip', 'val.zip', 'test.zip'):
            jobs.append(('bdd100k-' + item['path'], f'https://huggingface.co/datasets/{bdd_repo}/resolve/main/{item["path"]}?download=true', 'BDD100K', item['size'], 'sha256', item['lfs']['oid']))
    for split, size in [('train',1549875511),('val',81638851),('test-dev',311251787),('test-challenge',306200141)]:
        name = f'VisDrone2019-DET-{split}.zip'
        jobs.append((name, 'https://github.com/ultralytics/assets/releases/download/v0.0.0/' + name, 'VisDrone2019', size, None, None))
    jobs.append(('Vehicle_Tracking.zip','https://zenodo.org/api/records/18195750/files/Vehicle_Tracking.zip/content','VnTraffic',39489618,'md5','a1a511436e55b8168928a6a278ae2ba7'))
    (ROOT / 'download_sources.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    fetch_text('https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst', ROOT / 'BDD100K' / 'ORIGINAL_LICENSE.rst')
    fetch_text('https://raw.githubusercontent.com/VisDrone/VisDrone-Dataset/master/README.md', ROOT / 'VisDrone2019' / 'ORIGINAL_README.md')
    (ROOT / 'VisDrone2019' / 'LICENSE_SOURCE.txt').write_text('Official policy: https://aiskyeye.com/data-protection/\nAcademic use only. Policy lists CC BY-NC-SA 3.0 and names VisDrone2021; confirm version-specific terms before uses beyond classroom research.\n', encoding='utf-8')
    fetch_text('https://zenodo.org/api/records/18195750', ROOT / 'VnTraffic' / 'zenodo_metadata.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(download, jobs))
    print('ALL DOWNLOADS VERIFIED AND EXTRACTED', flush=True)

if __name__ == '__main__':
    main()
