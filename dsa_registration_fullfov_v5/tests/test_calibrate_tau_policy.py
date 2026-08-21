import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_tau_calibration_excludes_invalid_registration(tmp_path: Path):
    root = tmp_path / 'out'
    for patient, valid, value in [('1', 1, .02), ('2', 0, .80)]:
        case = root / 'Train' / patient / 'series'
        case.mkdir(parents=True)
        np.savez_compressed(
            case / 'change_maps.npz',
            canonical_logjac=np.full((8, 8), value, np.float32),
            stable=np.ones((8, 8), np.uint8),
            canonical_valid=np.ones((8, 8), np.uint8),
        )
        (case / 'features.json').write_text(json.dumps({'registration_valid': valid}))
    out = tmp_path / 'tau.json'
    script = Path(__file__).parents[1] / 'scripts' / 'calibrate_tau.py'
    subprocess.run([
        sys.executable, str(script), '--output-root', str(root), '--split', 'Train',
        '--quantile', '.95', '--out', str(out),
    ], check=True)
    result = json.loads(out.read_text())
    assert result['n_series'] == 1
    assert result['skipped_invalid_series'] == 1
    assert np.isclose(result['tau'], .02)
