#!/usr/bin/env python3
"""
Scan results/*/*.json to detect truncated/corrupt JSON files and optionally
move them to results/<dataset>/corrupt/. Also inject `experiment_id` into
valid JSON files when missing by parsing the filename (pattern: _exp<id>_results.json).

Usage:
  python3 scripts/fix_truncated_results.py --dry-run
  python3 scripts/fix_truncated_results.py --apply
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

PAT = re.compile(r'_exp(?P<id>[^_]+)_results\.json$')


def safe_write_json(path, data):
	# atomic write
	tmp = str(path) + '.tmp'
	with open(tmp, 'w') as f:
		json.dump(data, f, indent=2)
		f.flush()
		os.fsync(f.fileno())
	shutil.move(tmp, str(path))


def process_results(dry_run=True):
	root = Path('results')
	if not root.exists():
		print('No results/ directory found')
		return 0
	moved = []
	fixed = []
	for dataset_dir in sorted(root.iterdir()):
		if not dataset_dir.is_dir():
			continue
		for p in sorted(dataset_dir.glob('*.json')):
			fname = p.name
			try:
				with open(p, 'r') as f:
					data = json.load(f)
			except Exception as e:
				# corrupted/truncated
				corrupt_dir = dataset_dir / 'corrupt'
				corrupt_dir.mkdir(parents=True, exist_ok=True)
				corrupt_path = corrupt_dir / (fname + '.corrupt')
				print(f'CORRUPT: {p} -> {corrupt_path} ({e})')
				if not dry_run:
					shutil.move(str(p), str(corrupt_path))
					moved.append((str(p), str(corrupt_path)))
				continue
			# parsed OK
			if 'experiment_id' not in data or data.get('experiment_id') in (None, ''):
				m = PAT.search(fname)
				if m:
					exp_id = m.group('id')
					print(f'MISSING experiment_id in {p}; parsed id={exp_id}')
					if not dry_run:
						data['experiment_id'] = exp_id
						try:
							safe_write_json(p, data)
							fixed.append(str(p))
						except Exception as e:
							print(f'Failed to write {p}: {e}')
				else:
					print(f'No exp id in filename to extract for {p}')
			# otherwise valid and containing experiment_id
	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--apply', action='store_true', help='Apply changes (move corrupt files and write experiment_id)')
	args = parser.parse_args()
	if args.apply:
		print('Running fixer: APPLY mode')
		process_results(dry_run=False)
	else:
		print('Dry-run: no files will be modified. Use --apply to apply changes.')
		process_results(dry_run=True)
