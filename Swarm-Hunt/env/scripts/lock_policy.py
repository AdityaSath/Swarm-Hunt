"""Lock a policy (scripted or SB3 archive) into `env/models/` for reproducible evaluation.

Usage:
  PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/lock_policy.py --name scripted_v1 --source scripted
  PYTHONPATH="$(pwd)/env" .venv/bin/python env/scripts/lock_policy.py --name ppo_long --source ./sb3_logs_retrain_long/ppo_central.zip
"""
import argparse
from locked_policy import save_locked_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', required=True, help="'scripted' or path to SB3 .zip")
    args = parser.parse_args()
    manifest = save_locked_policy(args.name, args.source)
    print(f'Locked policy manifest saved to {manifest}')


if __name__ == '__main__':
    main()
