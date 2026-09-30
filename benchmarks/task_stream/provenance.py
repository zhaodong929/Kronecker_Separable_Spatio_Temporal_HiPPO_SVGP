"""Source and feature identities shared by selection, final runs and profiling."""
import hashlib
from pathlib import Path
import re
import subprocess


def source_revision(root):
    root = Path(root)
    release = root/'SOURCE_COMMIT'
    value = release.read_text().strip() if release.exists() else subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('Invalid source commit record')
    return value


def source_identity(root):
    root = Path(root)
    digest = hashlib.sha256()
    for folder in ('benchmarks/task_stream', 'benchmarks/three_domain', 'scripts', 'stvgp_kronecker', 'baselines'):
        for path in sorted((root/folder).rglob('*.py')):
            relative = path.relative_to(root)
            if 'external' in relative.parts or path.is_symlink():
                continue
            digest.update(str(relative).encode()+b'\0')
            digest.update(path.read_bytes())
    return digest.hexdigest()


def feature_identity(times, features):
    import numpy as np
    digest = hashlib.sha256()
    for array in (np.asarray(times), np.asarray(features)):
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
