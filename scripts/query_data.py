#!/usr/bin/env python3
"""
Optimized Blockchain Data Query Tool
====================================

High-performance script for generating blockchain data CSV files with:
- Batched RPC calls for maximum efficiency
- Pool address caching to minimize subgraph queries
- APR and price data collection in single operations
- Real-time performance monitoring

Output Format:
- block, timestamp, {vault}_bgt_apr, {token}_price_in_bera, {token}_pool_id
- Each LST token gets price and pool_id columns
- Data saved to data/ directory for downstream analysis

Performance Features:
- Pool cache refreshes every 45,000 blocks (configurable)
- Batch contract calls reduce RPC overhead
- Subgraph response time monitoring
- Progress tracking with ETA calculations
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.blockchain.optimized_batch_collector import OptimizedBatchCollector
from src.utils.vault_utils import get_all_vault_stake_tokens


def setup_logger(verbose: bool = False) -> None:
    """Configure logging with specified verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=config.LOG_FORMAT)


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


def _build_csv_headers(vault_info_list: list) -> list[str]:
    """Build CSV header row with vault APR and token price/pool columns."""
    headers = ["block", "timestamp"]

    # Add vault APR columns
    for info in vault_info_list:
        headers.append(f"{info['stake_token']['symbol']}_bgt_apr")

    # Add LST token price and pool_id columns
    for token in config.LST_TOKENS:
        headers.append(f"{token['symbol']}_price_in_bera")
        headers.append(f"{token['symbol']}_pool_id")

    return headers


def query_blockchain_data(
    start_block: int,
    end_block: int,
    interval: int,
    csv_path: Optional[Path] = None,
) -> None:
    """
    Query optimized blockchain data and save to CSV with batched RPC calls.

    Uses OptimizedBatchCollector for high-performance data collection:
    - Caches pool addresses to reduce subgraph queries
    - Batches contract calls for APR and price data
    - Monitors performance with timing metrics

    Args:
        start_block: Starting block number
        end_block: Ending block number
        interval: Block interval between data points
        csv_path: Output CSV file path (auto-generated if None)

    Raises:
        ValueError: If no valid vault information found
    """
    total_blocks = (end_block - start_block) // interval + 1
    start_time = time.time()

    logging.info("🚀 Starting optimized data generation:")
    logging.info(
        f"   📊 Range: {start_block:,} → {end_block:,} (interval: {interval:,})"
    )
    logging.info(f"   🎯 Target: {total_blocks} data points")

    # Initialize optimized batch collector
    batch_collector = OptimizedBatchCollector(config.ALCHEMY_API_KEY, config.NETWORK)

    # Get vault information
    vault_info_list = get_all_vault_stake_tokens(config.ALCHEMY_API_KEY, config.NETWORK)
    if not vault_info_list:
        raise ValueError("No valid vault information found")

    logging.info(f"💰 Found {len(vault_info_list)} vaults:")
    for info in vault_info_list:
        logging.info(f"   - {info['stake_token']['symbol']}: {info['vault_address']}")

    # Setup output path
    if csv_path is None:
        csv_path = Path("data/data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build CSV headers
    headers = _build_csv_headers(vault_info_list)
    logging.info(f"📝 CSV columns: {len(headers)} ({', '.join(headers[:5])}...)")
    logging.info(f"💾 Output: {csv_path}")

    # Main data collection loop with mega-batch processing
    successful_rows = 0
    failed_blocks = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

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

    # Final summary with performance metrics
    total_time = time.time() - start_time
    avg_rate = successful_rows / total_time if total_time > 0 else 0

    logging.info("\n✅ Data generation completed!")
    logging.info(f"   📊 Rows written: {successful_rows}/{total_blocks}")
    logging.info(f"   ❌ Failed blocks: {failed_blocks}")
    logging.info(f"   ⏱️  Total time: {total_time:.1f}s")
    logging.info(f"   🚀 Average rate: {avg_rate:.2f} blocks/sec")
    logging.info(f"   💾 Output file: {csv_path}")


def cli() -> None:
    """
    Command-line interface for optimized blockchain data generation.

    Provides arguments for block range, interval, output path, and logging level.
    Includes helpful examples and validation for user inputs.
    """
    parser = argparse.ArgumentParser(
        description="🚀 Generate optimized blockchain data CSV with batched RPC calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 data points with 50k block intervals
  python scripts/query_data.py --start 2900000 --end 5350000 --interval 50000

  # Custom output path with verbose logging
  python scripts/query_data.py --start 3000000 --end 3100000 --interval 10000 --csv my_data.csv -v

Performance Notes:
  - Pool cache refreshes every 45,000 blocks
  - Mega-batch processing: 20 blocks per batch (120 RPC calls)
  - Progress updates every 40 blocks with ETA calculations
        """,
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting block number for data collection",
    )
    parser.add_argument(
        "--end", type=int, required=True, help="Ending block number for data collection"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Block interval between data points (default: 1000)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output CSV file path (default: data/data.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging with debug information",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.start >= args.end:
        parser.error("Start block must be less than end block")
    if args.interval <= 0:
        parser.error("Interval must be positive")

    setup_logger(args.verbose)

    try:
        query_blockchain_data(
            start_block=args.start,
            end_block=args.end,
            interval=args.interval,
            csv_path=args.csv,
        )
    except KeyboardInterrupt:
        logging.info("\n🛑 Generation interrupted by user")
    except Exception as e:
        logging.error(f"❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
