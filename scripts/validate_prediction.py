#!/usr/bin/env python3
"""
Validate Prediction - Check prediction accuracy against actual results.

Usage:
    python scripts/validate_prediction.py 2026-01-30 7 15 22 31 35
    python scripts/validate_prediction.py 2026-01-30 --qvs 7,15,22,31,35
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def load_prediction(date_str):
    """Load prediction file for a given date."""
    pred_path = Path(f'data/predictions/daily/prediction_{date_str}.json')

    if not pred_path.exists():
        raise FileNotFoundError(f"No prediction found for {date_str}")

    with open(pred_path, 'r') as f:
        return json.load(f), pred_path


def validate_prediction(prediction, actual_qvs):
    """Validate prediction against actual results."""
    excluded_set = set(prediction['excluded_qvs'])
    actual_set = set(actual_qvs)

    # False positives: actual winners that were incorrectly excluded
    false_positives = sorted(actual_set & excluded_set)
    fp_count = len(false_positives)

    # Determine result category
    if fp_count == 0:
        result = "perfect"
    elif fp_count == 1:
        result = "good"
    elif fp_count <= 3:
        result = "fair"
    else:
        result = "poor"

    # Correctly predicted (actual winners in remaining candidates)
    correct = sorted(actual_set - excluded_set)

    return {
        'actual_qvs': sorted(actual_qvs),
        'false_positives': false_positives,
        'correct_predictions': correct,
        'fp_count': fp_count,
        'validated_at': datetime.now().isoformat(),
        'result': result
    }


def main():
    parser = argparse.ArgumentParser(description='Validate C5 prediction against actual results')
    parser.add_argument('date', help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('qvs', nargs='*', type=int, help='Actual winning QVs (5 numbers)')
    parser.add_argument('--qvs-str', type=str, help='Actual QVs as comma-separated string')

    args = parser.parse_args()

    # Parse QVs
    if args.qvs_str:
        actual_qvs = [int(x.strip()) for x in args.qvs_str.split(',')]
    elif args.qvs:
        actual_qvs = args.qvs
    else:
        # Interactive mode
        print(f"Validating prediction for: {args.date}")
        qvs_input = input("Enter the 5 actual winning QVs (space or comma separated): ")
        actual_qvs = [int(x.strip()) for x in qvs_input.replace(',', ' ').split()]

    # Validate input
    if len(actual_qvs) != 5:
        print(f"Error: Expected 5 QVs, got {len(actual_qvs)}")
        return 1

    if not all(1 <= qv <= 39 for qv in actual_qvs):
        print("Error: All QVs must be between 1 and 39")
        return 1

    if len(set(actual_qvs)) != 5:
        print("Error: QVs must be unique")
        return 1

    print()
    print('=' * 60)
    print(f'VALIDATING PREDICTION FOR {args.date}')
    print('=' * 60)
    print()

    # Load prediction
    try:
        prediction, pred_path = load_prediction(args.date)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Prediction file: {pred_path}")
    print(f"Actual QVs: {sorted(actual_qvs)}")
    print()

    # Validate
    validation = validate_prediction(prediction, actual_qvs)

    # Display results
    print('## Results')
    print()
    print(f"Excluded QVs (20):     {prediction['excluded_qvs']}")
    print(f"Actual winners (5):    {validation['actual_qvs']}")
    print()

    if validation['fp_count'] == 0:
        print("FALSE POSITIVES:       NONE - PERFECT PREDICTION!")
    else:
        print(f"FALSE POSITIVES ({validation['fp_count']}):   {validation['false_positives']}")
        print(f"                       (These winners were incorrectly excluded)")

    print()
    print(f"Correctly in candidates: {validation['correct_predictions']}")
    print()

    # Result summary
    result_emoji = {
        'perfect': '🎯 PERFECT',
        'good': '✓ GOOD',
        'fair': '○ FAIR',
        'poor': '✗ POOR'
    }

    print('-' * 60)
    print(f"FP@20: {validation['fp_count']}/5")
    print(f"Result: {result_emoji.get(validation['result'], validation['result'])}")
    print('-' * 60)
    print()

    # Update prediction file
    prediction['validation'] = validation

    with open(pred_path, 'w') as f:
        json.dump(prediction, f, indent=2)

    print(f"Updated: {pred_path}")
    print()

    # Show git command
    print("To commit this validation:")
    print(f"  git add {pred_path}")
    print(f'  git commit -m "Validate prediction for {args.date}: FP={validation["fp_count"]} ({validation["result"]})"')
    print("  git push")

    return 0


if __name__ == '__main__':
    exit(main())
