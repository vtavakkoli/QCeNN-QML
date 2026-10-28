"""Broader text snapshots retain content identity, row provenance and license."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from qcenn_qml import prepare_text


def test_public_text_snapshots_are_reproducible_and_tampering_is_rejected(monkeypatch, tmp_path):
    def rows(url):
        start = int(parse_qs(urlparse(url).query)['offset'][0])
        return {'rows': [{'row_idx': start+i, 'row': {'text': f'Public encyclopedia paragraph number {start+i} contains several words.'},
            'truncated_cells': []} for i in range(100)]}
    monkeypatch.setattr(prepare_text, 'download', rows)
    files = prepare_text.prepare(tmp_path / 'text')
    assert len(files) == 3 and all(Path(p).exists() for p in files)
    source = json.loads(Path(files[0]).with_suffix('.source.json').read_text())
    assert source['license'] == ['CC-BY-SA-3.0', 'GFDL']
    assert source['code_license_does_not_replace_dataset_license']
    monkeypatch.setattr(prepare_text, 'download', lambda _: pytest.fail('Immutable cache was redownloaded'))
    assert prepare_text.prepare(tmp_path / 'text') == files
    Path(files[0]).write_text('changed')
    with pytest.raises(ValueError, match='snapshot changed'):
        prepare_text.prepare(tmp_path / 'text')
