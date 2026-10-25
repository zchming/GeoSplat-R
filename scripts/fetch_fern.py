"""Fetch only the frozen fern 3-view split through ZIP HTTP ranges (about 2 MB)."""
import argparse
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def fetch_range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError('Server ignored range; refusing a full archive download')
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise RuntimeError('Incorrect range size')
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--archive-bytes', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tail = fetch_range(args.url, args.archive_bytes - 65536, args.archive_bytes - 1)
    end = struct.unpack_from('<4s4H2IH', tail, tail.rfind(b'PK\x05\x06'))
    size, offset = end[5:7]
    cd = fetch_range(args.url, offset, offset + size - 1)
    entries, cursor = {}, 0
    while cursor < len(cd):
        h = struct.unpack_from('<4s6H3I5H2I', cd, cursor)
        assert h[0] == b'PK\x01\x02'
        nl, xl, cl = h[10:13]
        name = cd[cursor+46:cursor+46+nl].decode()
        entries[name] = {'crc32': h[7], 'compressed': h[8], 'bytes': h[9],
                         'offset': h[-1], 'method': h[4]}
        cursor += 46 + nl + xl + cl
    prefix = 'nerf_llff_data/fern/'
    names = sorted(x[len(prefix+'images_8/'):] for x in entries
                   if x.startswith(prefix+'images_8/') and x.endswith('.png'))
    assert len(names) == 20
    train_ids, test_ids = [1, 10, 19], [0, 8, 16]
    targets = [('poses_bounds.npy', None)]
    targets += [('images_8/'+names[i], 'train') for i in train_ids]
    targets += [('images_8/'+names[i], 'test') for i in test_ids]
    records = []
    for relative, split in targets:
        name = prefix + relative
        e = entries[name]
        header = fetch_range(args.url, e['offset'], e['offset'] + 29)
        lh = struct.unpack('<4s5H3I2H', header)
        assert lh[0] == b'PK\x03\x04' and lh[3] == e['method']
        start = e['offset'] + 30 + lh[-2] + lh[-1]
        data = fetch_range(args.url, start, start + e['compressed'] - 1)
        raw = zlib.decompress(data, -15) if e['method'] == 8 else data
        assert len(raw) == e['bytes'] and zlib.crc32(raw) == e['crc32']
        target = args.output / (f'{split}/images/{Path(relative).name}' if split else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        records.append({'archive_member': name, 'local': str(target.relative_to(args.output)),
                        'sha256': hashlib.sha256(raw).hexdigest(), **e})
    manifest = {'url': args.url, 'archive_bytes': args.archive_bytes,
                'all_image_names': names, 'train_indices': train_ids, 'test_indices': test_ids,
                'unused_indices': [i for i in range(20) if i not in train_ids+test_ids],
                'downloaded_compressed_bytes': sum(x['compressed'] for x in records),
                'files': records, 'protocol': 'calibrated cameras; full-view SfM points never downloaded'}
    (args.output/'download.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
