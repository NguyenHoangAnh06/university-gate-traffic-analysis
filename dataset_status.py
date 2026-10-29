"""Print and save download progress without opening image contents."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'data'

def main():
    rows = []
    for manifest_name, archive_root in [('download_sources.json', ROOT / '_archives/detection'), ('tracking_download_sources.json', Path('D:/deeplearning-datasets/_downloads'))]:
        manifest = ROOT / manifest_name
        if not manifest.exists():
            continue
        for name, url, folder, size, algorithm, checksum in json.loads(manifest.read_text()):
            destination = ROOT / folder
            marker = destination / (name + '.verified.json')
            ledger = archive_root / (name + '.ranges.json')
            complete = marker.exists()
            downloaded = size if complete else 0
            if ledger.exists() and not complete:
                chunks = json.loads(ledger.read_text())
                downloaded = sum(min(16*1024*1024, size-i*16*1024*1024) for i in chunks)
            rows.append({'archive': name, 'verified_and_extracted': complete, 'downloaded_bytes': downloaded, 'total_bytes': size, 'percent': round(100*downloaded/size, 1), 'destination': str(destination)})
    status = {'complete': bool(rows) and all(r['verified_and_extracted'] for r in rows), 'verified_archives': sum(r['verified_and_extracted'] for r in rows), 'archive_count': len(rows), 'downloaded_GB': round(sum(r['downloaded_bytes'] for r in rows)/1e9, 2), 'total_GB': round(sum(r['total_bytes'] for r in rows)/1e9, 2), 'files': rows}
    (ROOT / 'download_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    for row in rows:
        print(('DONE' if row['verified_and_extracted'] else f'{row["percent"]:5.1f}%'), row['archive'])
    print(f'{status["verified_archives"]}/{status["archive_count"]} verified; {status["downloaded_GB"]}/{status["total_GB"]} GB downloaded')

if __name__ == '__main__':
    main()
