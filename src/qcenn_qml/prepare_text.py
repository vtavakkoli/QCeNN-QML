"""Archive public WikiText row slices for broader, reproducible text diagnostics.

This uses the public Hugging Face dataset-viewer API, needs no account, and
freezes every response and derived corpus by SHA256. Slices are all from the
same dataset, not three independently collected language benchmarks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .recurrent_mnist import write_json
from .resources import single_writer


DATASET = 'Salesforce/wikitext'
CONFIG = 'wikitext-2-raw-v1'
SOURCE = 'https://huggingface.co/datasets/Salesforce/wikitext'
SLICES = [(0, 1000), (12000, 1000), (24000, 1000)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'QCeNN-QML research corpus preparation'})
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


@single_writer
def prepare(output_dir):
    root = Path(output_dir).resolve()
    manifest = root / 'corpus-manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for name, checksum in previous['sha256'].items():
            if sha(root / name) != checksum:
                raise ValueError('Text snapshot changed: ' + name)
        return [str(root / name) for name in previous['corpus_files']]
    root.mkdir(parents=True, exist_ok=True)
    requests = []
    for offset, length in SLICES:
        for start in range(offset, offset + length, 100):
            query = urlencode({'dataset': DATASET, 'config': CONFIG, 'split': 'train', 'offset': start, 'length': 100})
            requests.append((start, 'https://datasets-server.huggingface.co/rows?' + query))
    def fetch(item):
        start, url = item
        path = root / f'rows-{start:06d}.json'
        if not path.exists():
            write_json(path, {'request_url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'response': download(url)})
        return start, json.loads(path.read_text())
    with ThreadPoolExecutor(max_workers=4) as executor:
        pages = dict(executor.map(fetch, requests))
    files = []
    for index, (offset, length) in enumerate(SLICES):
        paragraphs, indices, skipped = [], [], []
        for start in range(offset, offset + length, 100):
            for item in pages[start]['response']['rows']:
                value = item['row']['text'].strip()
                if item.get('truncated_cells'):
                    skipped.append(item['row_idx']); continue
                if value and not value.startswith('='):
                    paragraphs.append(value); indices.append(item['row_idx'])
        if len(paragraphs) < 20:
            raise ValueError('Public text API returned too few complete paragraphs')
        path = root / f'wikitext-slice-{index}.txt'
        path.write_text('\n'.join(paragraphs) + '\n', encoding='utf8')
        attribution = {'dataset': DATASET, 'configuration': CONFIG, 'upstream_split': 'train',
            'source_url': SOURCE, 'article_authors': 'Wikipedia contributors; see the upstream dataset and its original article sources',
            'dataset_authors': 'Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher',
            'citation': 'Pointer Sentinel Mixture Models, arXiv:1609.07843 (2016)',
            'license': ['CC-BY-SA-3.0', 'GFDL'],
            'license_urls': ['https://creativecommons.org/licenses/by-sa/3.0/', 'https://www.gnu.org/licenses/fdl-1.3.html'],
            'code_license_does_not_replace_dataset_license': True,
            'rows_requested': [offset, offset + length], 'retained_row_indices': indices,
            'skipped_truncated_rows': skipped, 'corpus_sha256': sha(path),
            'snapshot_basis': 'Archived dataset-viewer responses and content hashes, not a claimed pinned upstream revision',
            'evaluation_protocol': 'Existing seeded whole-sentence split before windows; this is not canonical WikiText perplexity',
            'independence': 'Three disjoint row slices of ONE dataset; do not treat these as independently collected corpora'}
        write_json(path.with_suffix('.source.json'), attribution)
        files.append(path.name)
    checksums = {p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != manifest.name}
    write_json(manifest, {'dataset': DATASET, 'corpus_files': files, 'sha256': checksums})
    return [str(root / name) for name in files]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    paths = prepare(parser.parse_args().output_dir)
    for path in paths:
        print(path)
