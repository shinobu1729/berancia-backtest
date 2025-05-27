#!/usr/bin/env python3
"""
Data validation script for blockchain data collection.
Validates CSV data by comparing random samples with fresh individual RPC calls.
Only uses batch_call_read_function from blockchain_reader, all other functionality is self-contained.
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.blockchain.optimized_batch_collector import OptimizedBatchCollector


def get_fresh_data_for_block(
    collector: OptimizedBatchCollector, block_number: int
) -> List[str]:
    """
    Fetch fresh data for a specific block using the same logic as the data collection scripts.
    Returns data in the same format as CSV: [block, timestamp, apr, price1, pool_id1, price2, pool_id2, ...]
    """
    # Use the existing get_csv_row_data method - it returns exactly what we need
    raw_data = collector.get_csv_row_data(block_number)
    if not raw_data:
        raise RuntimeError(f"Failed to get data for block {block_number}")

    # Convert all values to strings to match CSV format
    return [str(item) for item in raw_data]


def read_csv_file(csv_path: str) -> tuple[List[str], List[List[str]]]:
    """Read CSV file and return headers and all rows."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
        return headers, rows
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")


def select_random_rows(
    rows: List[List[str]], count: int, seed: Optional[int] = None
) -> List[tuple[int, List[str]]]:
    """Select random rows from the dataset."""
    if seed is not None:
        random.seed(seed)

    if count >= len(rows):
        return [(i, rows[i]) for i in range(len(rows))]

    indices = random.sample(range(len(rows)), count)
    return [(i, rows[i]) for i in sorted(indices)]


def compare_values(
    csv_value: str, fresh_value: str, field_name: str
) -> tuple[bool, float, str]:
    """Compare two values with appropriate tolerance based on field type."""
    if field_name in ["block", "timestamp"]:
        # Exact match for integers
        try:
            csv_int = int(csv_value)
            fresh_int = int(fresh_value)
            matches = csv_int == fresh_int
            diff = abs(csv_int - fresh_int)
            return matches, diff, "exact"
        except ValueError:
            return False, float("inf"), "parse_error"

    elif "pool_id" in field_name:
        # Exact string match for pool IDs (addresses)
        matches = csv_value.lower() == fresh_value.lower()
        diff = 0.0 if matches else 1.0
        return matches, diff, "exact_string"

    elif "apr" in field_name:
        # Higher tolerance for APR (can fluctuate slightly)
        try:
            csv_float = float(csv_value)
            fresh_float = float(fresh_value)
            diff = abs(csv_float - fresh_float)
            tolerance = max(abs(csv_float) * 0.01, 1e-6)  # 1% or 1e-6 absolute
            matches = diff <= tolerance
            return matches, diff, f"apr_tolerance_{tolerance:.2e}"
        except ValueError:
            return False, float("inf"), "parse_error"

    else:
        # Price fields - moderate tolerance for floating point precision
        try:
            csv_float = float(csv_value)
            fresh_float = float(fresh_value)
            diff = abs(csv_float - fresh_float)
            tolerance = max(abs(csv_float) * 0.001, 1e-12)  # 0.1% or 1e-12 absolute
            matches = diff <= tolerance
            return matches, diff, f"price_tolerance_{tolerance:.2e}"
        except ValueError:
            return False, float("inf"), "parse_error"


def validate_single_row(
    row_index: int,
    csv_row: List[str],
    headers: List[str],
    collector: OptimizedBatchCollector,
) -> Dict[str, Any]:
    """Validate a single CSV row against fresh RPC data."""
    try:
        block_number = int(csv_row[0])
        logging.info(f"🔍 Validating row {row_index + 1}/? (block {block_number})")

        # Get fresh data
        fresh_data = get_fresh_data_for_block(collector, block_number)

        # Compare each field
        matches = {}
        discrepancies = []
        total_fields = len(headers)
        matched_fields = 0

        for header, csv_value, fresh_value in zip(headers, csv_row, fresh_data):
            is_match, diff, comparison_type = compare_values(
                csv_value, fresh_value, header
            )
            matches[header] = {
                "match": is_match,
                "csv_value": csv_value,
                "fresh_value": fresh_value,
                "difference": diff,
                "comparison_type": comparison_type,
            }

            if is_match:
                matched_fields += 1
            else:
                discrepancies.append(
                    {
                        "field": header,
                        "csv_value": csv_value,
                        "fresh_value": fresh_value,
                        "difference": diff,
                        "comparison_type": comparison_type,
                    }
                )

        match_percentage = (matched_fields / total_fields) * 100

        status = "✅" if match_percentage == 100.0 else "⚠️"
        logging.info(f"   {status} Block {block_number}: {match_percentage:.1f}% match")

        return {
            "success": True,
            "block_number": block_number,
            "match_percentage": match_percentage,
            "matches": matches,
            "discrepancies": discrepancies,
            "total_fields": total_fields,
            "matched_fields": matched_fields,
        }

    except Exception as e:
        logging.error(f"   ❌ Block {csv_row[0]}: Validation failed - {e}")
        return {
            "success": False,
            "block_number": int(csv_row[0]) if csv_row[0].isdigit() else None,
            "error": str(e),
        }


def validate_csv_data(
    csv_path: str, sample_count: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Main validation function."""
    logging.info("🔍 Starting CSV data validation:")
    logging.info(f"   📂 CSV file: {csv_path}")
    logging.info(f"   🎯 Sample count: {sample_count}")

    # Read CSV
    try:
        headers, rows = read_csv_file(csv_path)
        logging.info(f"   📊 Total rows in CSV: {len(rows)}")
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return {"success": False, "error": str(e)}

    # Select random rows
    selected_rows = select_random_rows(rows, sample_count, seed)
    logging.info(f"   🎲 Selected {len(selected_rows)} random rows for validation")

    # Initialize optimized batch collector
    collector = OptimizedBatchCollector(config.ALCHEMY_API_KEY, config.NETWORK)

    # Validate each selected row
    start_time = time.time()
    results = []
    successful_validations = 0
    total_match_percentage = 0.0

    for i, (_, row) in enumerate(selected_rows):
        result = validate_single_row(i, row, headers, collector)
        results.append(result)

        if result["success"]:
            successful_validations += 1
            total_match_percentage += result["match_percentage"]

    validation_time = time.time() - start_time

    # Calculate summary statistics
    average_match_percentage = (
        total_match_percentage / successful_validations
        if successful_validations > 0
        else 0.0
    )
    perfect_matches = sum(
        1 for r in results if r.get("success") and r.get("match_percentage") == 100.0
    )

    # Generate summary
    logging.info("\n📋 Validation Summary:")
    logging.info(
        f"   ✅ Successful validations: {successful_validations}/{len(selected_rows)}"
    )
    logging.info(
        f"   ❌ Failed validations: {len(selected_rows) - successful_validations}"
    )
    logging.info(
        f"   🎯 Perfect matches: {perfect_matches}/{successful_validations} ({perfect_matches/successful_validations*100:.1f}%)"
        if successful_validations > 0
        else "   🎯 Perfect matches: 0/0"
    )
    logging.info(f"   📊 Average match percentage: {average_match_percentage:.2f}%")
    logging.info(f"   ⏱️  Total validation time: {validation_time:.1f}s")

    # Report discrepancies
    all_discrepancies = []
    for result in results:
        if result.get("success") and result.get("discrepancies"):
            all_discrepancies.extend(result["discrepancies"])

    if all_discrepancies:
        logging.info(f"\n⚠️  Found {len(all_discrepancies)} discrepancies:")
        for disc in all_discrepancies[:10]:  # Show first 10
            logging.info(
                f"   • {disc['field']}: CSV={disc['csv_value']}, Fresh={disc['fresh_value']} (diff={disc['difference']:.2e})"
            )
        if len(all_discrepancies) > 10:
            logging.info(f"   ... and {len(all_discrepancies) - 10} more")
    else:
        logging.info("\n🎉 No discrepancies found! All data matches perfectly.")

    return {
        "success": True,
        "total_rows": len(rows),
        "sampled_rows": len(selected_rows),
        "successful_validations": successful_validations,
        "perfect_matches": perfect_matches,
        "average_match_percentage": average_match_percentage,
        "validation_time": validation_time,
        "results": results,
        "discrepancies": all_discrepancies,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate CSV data against fresh blockchain data"
    )
    parser.add_argument(
        "--csv", default="data/data.csv", help="Path to CSV file to validate"
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of random rows to validate"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    # Run validation
    result = validate_csv_data(args.csv, args.count, args.seed)

    if not result["success"]:
        logging.error(f"Validation failed: {result.get('error', 'Unknown error')}")
        exit(1)

    # Exit with appropriate code
    if result["perfect_matches"] == result["successful_validations"]:
        exit(0)  # All perfect matches
    else:
        exit(2)  # Some discrepancies found


if __name__ == "__main__":
    main()
