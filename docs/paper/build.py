"""Build three manuscript profiles from shared source and a verified full run.

Temporary layout fixtures use an internal API, are watermarked, and cannot
pass publication verification. No execution-only override exists in the CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'services/agent-rl/scripts'))
from paper_assets import write_assets
from publication_statistics import derive
from verify_release import digest, require, verify, verify_hashes

PROFILES = {'full': 'main.tex', 'core': 'core/main.tex', 'workshop': 'workshop.tex'}
SOURCE_NAMES = ('article.tex', 'main.tex', 'workshop.tex', 'core/main.tex',
                'references.bib', 'build.py', 'paper_assets.py')
ASSET_NAMES = ('metrics.tex', 'tables.tex', 'statistics.json', 'figures/calibration.png',
               'figures/state_interventions.png', 'figures/steering.png')


def load(path):
    return json.loads(path.read_text())


def input_statistics(run, corpus):
    evidence = verify(run, corpus)
    reports = run / 'reports'
    statistics = {'schema_version': 1, 'verification': evidence,
                  'input_manifest_sha256': digest(reports / 'run_manifest.json'),
                  'statistics_source_sha256': digest(ROOT / 'services/agent-rl/scripts/publication_statistics.py'),
                  **derive(reports)}
    context = {'manifest': load(reports / 'run_manifest.json'),
               'parity': load(reports / 'rollout_parity.json')}
    return statistics, context


def stage_sources(destination, *, fixture=False):
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        if name.endswith('.py'):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        contents = (HERE / name).read_text()
        if fixture and name in PROFILES.values():
            contents = '\\def\\layoutfixture{1}\n' + contents
        target.write_text(contents)


def render_profiles(stage, targets):
    require(shutil.which('tectonic') is not None, 'tectonic is required to compile the manuscripts')
    require(shutil.which('pdfinfo') is not None, 'pdfinfo is required to verify PDF page counts')
    profiles = {}
    for target in targets:
        tex = stage / PROFILES[target]
        result = subprocess.run(['tectonic', '--keep-logs', tex.name], cwd=tex.parent,
                                capture_output=True, text=True)
        log = tex.with_suffix('.log')
        messages = result.stdout + result.stderr + (log.read_text(errors='replace') if log.exists() else '')
        require(result.returncode == 0, f'LaTeX compilation failed for {target}:\n{messages[-9000:]}')
        final_log = log.read_text(errors='replace') if log.exists() else messages
        defects = re.findall(r'[^\n]*(?:Overfull \\[hv]box|[Cc]itation[^\n]*undefined|undefined references)[^\n]*', final_log)
        require(not defects, f'unresolved manuscript layout/citation defects in {target}: {defects}')
        pdf = tex.with_suffix('.pdf')
        info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True, text=True).stdout
        count = re.search(r'^Pages:\s+(\d+)', info, flags=re.M)
        require(count is not None and int(count[1]) > 0, f'invalid PDF for {target}')
        profiles[target] = {'pdf': str(pdf.relative_to(stage)), 'pages': int(count[1])}
    return profiles


def build_from_statistics(statistics, context, output, targets, *, fixture=False):
    require(targets and set(targets) <= set(PROFILES), 'unknown or empty manuscript profiles')
    require(statistics['verification']['scientific_budgets_checked'] is True or fixture,
            'manuscripts require completed scientific budgets')
    with tempfile.TemporaryDirectory(prefix='nano-l1-manuscript-') as temporary:
        stage = Path(temporary)
        stage_sources(stage, fixture=fixture)
        write_assets(statistics, context, stage, fixture=fixture)
        (stage / 'statistics.json').write_text(json.dumps(statistics, indent=2, allow_nan=False) + '\n')
        profiles = render_profiles(stage, targets)
        names = list(ASSET_NAMES)
        names += [row['pdf'] for row in profiles.values()]
        # Copy only after every requested profile compiles without defects.
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage / name, destination)
        manuscript_sources = {str((HERE / name).relative_to(ROOT)): digest(HERE / name)
                              for name in SOURCE_NAMES}
        manifest = {'schema_version': 1, 'profiles': profiles,
                    'scientific_budgets_checked': statistics['verification']['scientific_budgets_checked'],
                    'layout_fixture': fixture, 'input_manifest_sha256': statistics['input_manifest_sha256'],
                    'manuscript_source_sha256': manuscript_sources,
                    'artifacts_sha256': {name: digest(output / name) for name in names}}
        (output / 'manuscript_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_artifacts(output):
    manifest = load(output / 'manuscript_manifest.json')
    require(manifest['schema_version'] == 1, 'unsupported manuscript manifest')
    require(manifest['scientific_budgets_checked'] is True and manifest['layout_fixture'] is False,
            'layout or smoke artifacts cannot verify a publication')
    profiles = manifest['profiles']
    require(profiles and set(profiles) <= set(PROFILES), 'invalid manuscript profiles')
    for target, row in profiles.items():
        require(row['pdf'] == str(Path(PROFILES[target]).with_suffix('.pdf'))
                and isinstance(row['pages'], int) and not isinstance(row['pages'], bool)
                and row['pages'] > 0, f'invalid manuscript profile: {target}')
    require(set(manifest['manuscript_source_sha256']) ==
            {str((HERE / name).relative_to(ROOT)) for name in SOURCE_NAMES},
            'manuscript source membership differs')
    require(set(manifest['artifacts_sha256']) ==
            set(ASSET_NAMES) | {row['pdf'] for row in profiles.values()},
            'manuscript artifact membership differs')
    verify_hashes(ROOT, manifest['manuscript_source_sha256'])
    verify_hashes(output, manifest['artifacts_sha256'])
    return manifest


def verify_build(run, corpus, output):
    expected, _ = input_statistics(run, corpus)
    manifest = verify_artifacts(output)
    require(manifest['input_manifest_sha256'] == expected['input_manifest_sha256'],
            'manuscript and selected experiment manifests differ')
    require(load(output / 'statistics.json') == expected,
            'manuscript statistics differ from reconstructed experiment inputs')
    return manifest


def numerical_claims(statistics):
    return {key: value for key, value in statistics.items()
            if key not in {'verification', 'input_manifest_sha256', 'statistics_source_sha256'}}


def compare_claims(current, published, path='statistics'):
    """Compare a new complete run with the recorded numerical claim values."""
    if isinstance(published, dict):
        require(isinstance(current, dict) and current.keys() == published.keys(), f'claim keys differ: {path}')
        for key in published:
            compare_claims(current[key], published[key], f'{path}/{key}')
    elif isinstance(published, list):
        require(isinstance(current, list) and len(current) == len(published), f'claim count differs: {path}')
        for index, (a, b) in enumerate(zip(current, published, strict=True)):
            compare_claims(a, b, f'{path}/{index}')
    elif isinstance(published, (int, float)) and not isinstance(published, bool):
        require(isinstance(current, (int, float)) and not isinstance(current, bool)
                and abs(current - published) <= 1e-9,
                f'numerical claim differs: {path}')
    elif isinstance(published, bool):
        require(isinstance(current, bool) and current is published, f'claim differs: {path}')
    else:
        require(current == published, f'claim differs: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT / 'data/releases/research_v1')
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data/releases/market_snapshot_v1/corpus.npz')
    ap.add_argument('--out', type=Path, default=HERE)
    ap.add_argument('--targets', nargs='+', choices=tuple(PROFILES), default=list(PROFILES))
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    if args.verify_only:
        manifest = verify_build(args.run, args.corpus, args.out)
    else:
        statistics, context = input_statistics(args.run, args.corpus)
        manifest = build_from_statistics(statistics, context, args.out, args.targets)
        verify_build(args.run, args.corpus, args.out)
    print(json.dumps({'profiles': manifest['profiles'],
                      'scientific_budgets_checked': manifest['scientific_budgets_checked']}, indent=2))


if __name__ == '__main__':
    main()
