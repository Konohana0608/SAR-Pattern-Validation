"""
Runner script for measurement validation tests with smart rerun support.

Usage:
    # Run only missing cases (cases without artifacts)
    python run_measurement_validation_tests.py

    # Run all 5800 MHz cases and any missing cases
    python run_measurement_validation_tests.py --rerun 5800mhz

    # Run 2450 and 5800 MHz cases
    python run_measurement_validation_tests.py --rerun 2450mhz --rerun 5800mhz

    # Regenerate all artifacts (ignores existing results)
    python run_measurement_validation_tests.py --regenerate-artifacts

    # Run with plot generation
    python run_measurement_validation_tests.py --save-plots

    # Combine flags
    python run_measurement_validation_tests.py --rerun 5800mhz --save-plots
"""

import sys

from sar_pattern_validation.measurement_validation_report import main

if __name__ == "__main__":
    sys.exit(main())
