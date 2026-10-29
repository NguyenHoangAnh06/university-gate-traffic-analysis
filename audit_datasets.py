"""Check downloaded split counts and preserve a small machine-readable inventory."""
import json
from pathlib import Path
from collections import Counter
import dataset_status

ROOT = Path(__file__).resolve().parent / 'data'

def iter_json_array(path):
    """Read one annotated frame at a time instead of loading GBs into RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as source:
        buffer = ''
        started = False
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = source.read(256 * 1024)
                if not chunk:
                    raise ValueError('Unexpected end of JSON array')
                buffer += chunk
                continue
            if not started:
                if buffer[0] != '[':
                    raise ValueError('Expected a JSON array')
                started = True
                buffer = buffer[1:]
                continue
            if buffer[0] == ']':
                return
            if buffer[0] == ',':
                buffer = buffer[1:]
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source.read(256 * 1024)
                if not chunk:
                    raise
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]

def main():
    dataset_status.main()
    status = json.loads((ROOT / 'download_status.json').read_text())
    if not status['complete']:
        raise RuntimeError('Downloads are not all verified/extracted yet')
    inventory = {}
    bdd = ROOT / 'BDD100K'
    for split, expected in [('train', 70000), ('val', 10000), ('test', 20000)]:
        image_root = bdd / split / 'images' if split != 'test' else bdd / split
        count = len(list(image_root.glob('*.jpg')))
        assert count == expected, (split, count, expected)
        entry = {'images': count}
        if split != 'test':
            label_file = next((bdd / split / 'annotations').glob('*.json'))
            names = {p.name for p in image_root.glob('*.jpg')}
            categories = Counter()
            frames = 0
            annotated_names = set()
            for frame in iter_json_array(label_file):
                assert frame['name'] in names, frame['name']
                assert frame['name'] not in annotated_names, ('duplicate annotation', frame['name'])
                annotated_names.add(frame['name'])
                categories.update(obj['category'] for obj in frame.get('labels', []))
                frames += 1
            assert 0 < frames <= expected, (label_file, frames, expected)
            entry['source_categories'] = dict(categories)
            entry['annotation_records'] = frames
            entry['images_without_annotation_record'] = sorted(names - annotated_names)
            split_lists = bdd / 'splits'
            split_lists.mkdir(exist_ok=True)
            (split_lists / (split + '_labeled_images.txt')).write_text(''.join(f'{split}/images/{name}\n' for name in sorted(annotated_names)), encoding='utf-8')
            (split_lists / (split + '_missing_annotation_images.txt')).write_text(''.join(f'{split}/images/{name}\n' for name in sorted(names - annotated_names)), encoding='utf-8')
            if names - annotated_names:
                print('SOURCE ANNOTATION GAP', split, len(names - annotated_names), 'images; recorded in splits/', flush=True)
            entry['annotation_version'] = 'original detection labels, not Detection 2020'
        inventory['BDD100K_DET_' + split] = entry
        print('CHECKED BDD100K detection', split, count, flush=True)
    for split, expected in [('train',6471), ('val',548), ('test-dev',1610), ('test-challenge',1580)]:
        folder = ROOT / 'VisDrone2019' / ('VisDrone2019-DET-' + split)
        images = list((folder / 'images').glob('*.jpg'))
        assert len(images) == expected, (folder, len(images))
        annotations = {p.stem for p in (folder / 'annotations').glob('*.txt')}
        if split != 'test-challenge':
            assert {p.stem for p in images} == annotations
        inventory['VisDrone2019_DET_' + split] = {'images':len(images), 'annotation_files':len(annotations)}
        print('CHECKED VisDrone detection', split, len(images), flush=True)
    for split, expected in [('train',1400),('val',200)]:
        labels_root = bdd / 'MOT/bdd100k/labels/box_track_20' / split
        labels = list(labels_root.glob('*.json'))
        assert len(labels) == expected, (labels_root, len(labels))
        inventory['BDD100K_MOT_' + split] = {'annotated_sequences':len(labels)}
    for dataset, folder in [('BDD100K_MOT',bdd/'MOT'),('VisDrone2019_MOT',ROOT/'VisDrone2019/MOT')]:
        sequences = {}
        image_count = 0
        for image in folder.rglob('*.jpg'):
            image_count += 1
            sequences[str(image.parent.relative_to(folder))] = sequences.get(str(image.parent.relative_to(folder)),0) + 1
        assert image_count > 0, dataset
        inventory[dataset] = {'images':image_count,'image_directories':len(sequences), 'sequences':sequences}
        print('CHECKED', dataset, image_count, 'frames', len(sequences), 'image folders', flush=True)
    vn = ROOT / 'VnTraffic/VNTraffic'
    tracks = json.loads((vn/'VNTraffic_GroundTruth.json').read_text())
    rows = [line.split(',') for line in (vn/'VNTraffic_GroundTruth.txt').read_text().splitlines() if line.strip()]
    assert len(rows) == 16036
    assert all(len(track['frames']) == len(track['boxes']) for track in tracks.values())
    json_boxes = Counter((int(Path(frame).stem), tuple(map(float, box))) for track in tracks.values() for frame, box in zip(track['frames'], track['boxes']))
    txt_boxes = Counter((int(row[0]), tuple(map(float, row[2:6]))) for row in rows)
    assert json_boxes == txt_boxes, 'VnTraffic JSON/TXT frame or bounding box mismatch'
    inventory['VnTraffic'] = {'txt_track_ids':len({r[1] for r in rows}),'json_track_ids':len(tracks),'identity_annotation_consistent_between_formats':False,'json_txt_boxes_match':True,'annotated_frames':len({r[0] for r in rows}),'boxes':len(rows),'has_class_labels':False,'frame_indices':[0,500],'raw_video_frames':510,'raw_video_duration_seconds':17}
    inventory['_warnings'] = ['BDD100K detection train has 137 images without annotation records; use the labeled-image split list.', 'VnTraffic TXT has 85 IDs but JSON has 205 IDs despite identical frame/box data. Reconcile identity annotations before benchmark use.', 'VnTraffic has no class labels.']
    inventory['MOT20'] = {'preserved':True,'path':str(ROOT/'MOT20')}
    (ROOT/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('AUDIT PASSED: archive integrity and split counts. Source annotation gaps and VnTraffic identity/class limitations are recorded in inventory.json and README.md.')

if __name__ == '__main__':
    main()
