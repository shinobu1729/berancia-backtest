#!/usr/bin/env python3
"""
Blockchain Data Append Tool
===========================

Appends new blockchain data to existing CSV files by:
- Reading the last block number and detecting interval from existing data
- Continuing data collection from where it left off
- Appending new rows to the existing CSV file
- Maintaining data consistency and format

Features:
- Automatic last block detection from CSV
- Interval auto-detection from existing data
- Seamless continuation of data collection
- Same performance optimizations as query_data.py
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.blockchain.optimized_batch_collector import OptimizedBatchCollector
from src.utils.vault_utils import get_all_vault_stake_tokens

LOG_FORMAT = "[%(levelname)s] %(message)s"


def setup_logger(verbose: bool = False) -> None:
    """Configure logging with specified verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _calculate_progress_stats(
    current_block: int, start_block: int, end_block: int, start_time: float
) -> tuple[float, float]:
    """Calculate progress percentage and estimated time remaining."""
    total_blocks = end_block - start_block
    completed_blocks = current_block - start_block

    progress = (completed_blocks / total_blocks) * 100 if total_blocks > 0 else 0
    elapsed = time.time() - start_time

    if completed_blocks > 0:
        eta = (elapsed / completed_blocks) * (total_blocks - completed_blocks)
    else:
        eta = 0

    return progress, eta


def _read_csv_info(csv_path: Path) -> Tuple[Optional[int], Optional[int], list]:
    """
    Read CSV file and extract last block, interval, and headers.

    Returns:
        Tuple of (last_block, interval, headers)
    """
    if not csv_path.exists():
        return None, None, []

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)

            # Read headers
            headers = next(reader, [])
            if not headers:
                return None, None, []

            # Read all data rows to find last block and detect interval
            rows = list(reader)
            if len(rows) == 0:
                return None, None, headers

            # Get last block
            last_row = rows[-1]
            last_block = int(last_row[0]) if last_row[0].isdigit() else None

            # Detect interval from last few rows
            interval = None
            if len(rows) >= 2:
                # Use the difference between last two blocks as interval
                second_last_block = int(rows[-2][0]) if rows[-2][0].isdigit() else None
                if last_block and second_last_block:
                    interval = last_block - second_last_block

            return last_block, interval, headers

    except Exception as e:
        logging.error(f"Failed to read CSV file {csv_path}: {e}")
        return None, None, []


def _validate_csv_headers(csv_headers: list, expected_headers: list) -> bool:
    """Validate that CSV headers match expected format."""
    if len(csv_headers) != len(expected_headers):
        return False

    for csv_h, exp_h in zip(csv_headers, expected_headers):
        if csv_h != exp_h:
            return False

    return True


def append_blockchain_data(
    end_block: int,
    csv_path: Optional[Path] = None,
    interval: Optional[int] = None,
) -> None:
    """
    Append new blockchain data to existing CSV file.

    Args:
        end_block: Target ending block number
        csv_path: Path to existing CSV file (default: data/data.csv)
        interval: Block interval override (auto-detected if None)

    Raises:
        ValueError: If CSV file doesn't exist or has invalid format
        FileNotFoundError: If CSV file doesn't exist
    """
    # Setup CSV path
    if csv_path is None:
        csv_path = Path("data/data.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read existing CSV info
    last_block, detected_interval, csv_headers = _read_csv_info(csv_path)

    if last_block is None:
        raise ValueError(f"Could not determine last block from CSV: {csv_path}")

    # Use provided interval or detected interval
    if interval is None:
        if detected_interval is None:
            raise ValueError("Could not detect interval from CSV and none provided")
        interval = detected_interval

    # Calculate start block (next block to collect)
    start_block = last_block + interval

    if start_block > end_block:
        logging.info(
            f"No new blocks to collect. Last block: {last_block}, Target: {end_block}"
        )
        return

    # Initialize components
    start_time = time.time()
    total_blocks = (end_block - start_block) // interval + 1

    logging.info("📝 Starting data append operation:")
    logging.info(f"   📂 CSV file: {csv_path}")
    logging.info(f"   📊 Last block in CSV: {last_block:,}")
    logging.info(f"   🔢 Detected interval: {interval:,}")
    logging.info(f"   📈 New range: {start_block:,} → {end_block:,}")
    logging.info(f"   🎯 New data points: {total_blocks}")

    # Initialize batch collector
    batch_collector = OptimizedBatchCollector(config.ALCHEMY_API_KEY, config.NETWORK)

    # Get vault information and validate headers
    vault_info_list = get_all_vault_stake_tokens(config.ALCHEMY_API_KEY, config.NETWORK)
    if not vault_info_list:
        raise ValueError("No valid vault information found")

    # Build expected headers and validate
    expected_headers = ["block", "timestamp"]
    for info in vault_info_list:
        expected_headers.append(f"{info['stake_token']['symbol']}_bgt_apr")
    for token in config.LST_TOKENS:
        expected_headers.append(f"{token['symbol']}_price_in_bera")
        expected_headers.append(f"{token['symbol']}_pool_id")

    if not _validate_csv_headers(csv_headers, expected_headers):
        logging.warning("CSV headers don't match expected format")
        logging.warning(f"CSV headers: {csv_headers}")
        logging.warning(f"Expected: {expected_headers}")

    logging.info(f"💰 Found {len(vault_info_list)} vaults:")
    for info in vault_info_list:
        logging.info(f"   - {info['stake_token']['symbol']}: {info['vault_address']}")

    # Main data collection loop with mega-batch processing
    successful_rows = 0
    failed_blocks = 0

    # Open CSV in append mode
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Batch CSV writing optimization
        batch_buffer = []  # Buffer to accumulate rows before writing
        BUFFER_SIZE = 500  # Write every 500 rows for optimal I/O performance

        current_block = start_block
        block_index = 0

        while current_block <= end_block:
            # Prepare block numbers for mega-batch
            block_batch = []
            for i in range(config.BATCH_LOOP_COUNT):
                if current_block + (i * interval) <= end_block:
                    block_batch.append(current_block + (i * interval))
                else:
                    break

            if not block_batch:
                break

            try:
                # Collect data for multiple blocks using mega-batch
                rows_data = batch_collector.get_multiple_csv_rows_data(block_batch)

                # Accumulate rows in batch buffer
                for i, row_data in enumerate(rows_data):
                    block_index += 1

                    if row_data is not None:
                        batch_buffer.append(row_data)
                        successful_rows += 1

                        # Write batch when buffer is full
                        if len(batch_buffer) >= BUFFER_SIZE:
                            writer.writerows(batch_buffer)
                            csvfile.flush()
                            batch_buffer.clear()
                    else:
                        failed_blocks += 1
                        logging.warning(
                            f"❌ Failed to collect data for block {block_batch[i]}"
                        )

                # Progress reporting with performance metrics
                if (
                    block_index % (config.BATCH_LOOP_COUNT * 2) == 0
                    or current_block + (len(block_batch) * interval) > end_block
                ):
                    last_block = block_batch[-1] if block_batch else current_block
                    progress, eta = _calculate_progress_stats(
                        last_block, start_block, end_block, start_time
                    )
                    elapsed = time.time() - start_time
                    rate = successful_rows / elapsed if elapsed > 0 else 0

                    logging.info(
                        f"📈 Progress: {progress:.1f}% | "
                        f"Blocks: {successful_rows}/{total_blocks} | "
                        f"Rate: {rate:.1f} blk/s | "
                        f"ETA: {eta / 60:.1f}m"
                    )

            except KeyboardInterrupt:
                logging.info(f"\n🛑 Interrupted by user at block {current_block}")
                break
            except Exception as e:
                failed_blocks += len(block_batch)
                logging.error(f"❌ Error processing block batch {block_batch}: {e}")

            # Move to next batch
            current_block += len(block_batch) * interval

        # Write any remaining buffered rows
        if batch_buffer:
            writer.writerows(batch_buffer)
            csvfile.flush()
            batch_buffer.clear()

    # Final summary
    total_time = time.time() - start_time
    avg_rate = successful_rows / total_time if total_time > 0 else 0

    logging.info("\n✅ Data append completed!")
    logging.info(f"   📊 New rows appended: {successful_rows}/{total_blocks}")
    logging.info(f"   ❌ Failed blocks: {failed_blocks}")
    logging.info(f"   ⏱️  Total time: {total_time:.1f}s")
    logging.info(f"   🚀 Average rate: {avg_rate:.2f} blocks/sec")
    logging.info(f"   💾 Updated file: {csv_path}")


def cli() -> None:
    """
    Command-line interface for appending blockchain data to existing CSV files.
    """
    parser = argparse.ArgumentParser(
        description="📝 Append new blockchain data to existing CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Append data up to block 5500000 (auto-detect interval)
  python scripts/append_data.py --end 5500000

  # Append with custom interval
  python scripts/append_data.py --end 5500000 --interval 1000

  # Append to specific CSV file
  python scripts/append_data.py --end 5500000 --csv my_data.csv -v

Notes:
  - Automatically detects last block and interval from existing CSV
  - Uses mega-batch processing: 20 blocks per batch (120 RPC calls)
  - Maintains data consistency and format
        """,
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Target ending block number for data collection",
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Block interval override (auto-detected from CSV if not provided)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file path to append to (default: data/data.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging with debug information",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.interval is not None and args.interval <= 0:
        parser.error("Interval must be positive")

    setup_logger(args.verbose)

    try:
        append_blockchain_data(
            end_block=args.end,
            csv_path=args.csv,
            interval=args.interval,
        )
    except KeyboardInterrupt:
        logging.info("\n🛑 Append operation interrupted by user")
    except Exception as e:
        logging.error(f"❌ Append operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
