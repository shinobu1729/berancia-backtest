#!/usr/bin/env python3
"""
Backtest Script for LST Token Strategy
=====================================

Executes backtests on generated CSV data to simulate LST token routing strategies
with configurable compound intervals and routing modes.
"""

import argparse
import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config


def parse_interval(interval_str: str) -> int:
    """Parse compound interval string to seconds."""
    if interval_str == "0m":
        return 0

    match = re.match(r"(\d+)([mhd])", interval_str.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}")

    value, unit = int(match.group(1)), match.group(2)
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def get_route_symbol(row_data: Dict[str, str], mode: str) -> str:
    """Determine routing symbol based on backtest mode."""
    if mode == "BGT":
        # BGT mode doesn't route to any specific token
        return "BGT"
    
    if mode != "auto":
        valid_symbols = [token["symbol"] for token in config.LST_TOKENS]
        if mode not in valid_symbols:
            raise ValueError(f"Invalid routing symbol: {mode}")
        return mode

    # Auto mode - find highest price symbol
    highest_price = 0.0
    best_symbol = config.LST_TOKENS[0]["symbol"]

    for token in config.LST_TOKENS:
        symbol = token["symbol"]
        price_key = f"{symbol}_price_in_bera"

        if price_key in row_data:
            try:
                price = float(row_data[price_key])
                if price > highest_price:
                    highest_price = price
                    best_symbol = symbol
            except (ValueError, TypeError):
                continue

    return best_symbol


def should_compound(current_ts: int, last_compound_ts: int, interval: int) -> bool:
    """Check if compounding should occur based on time elapsed."""
    if interval == 0:
        return True
    return (current_ts - last_compound_ts) >= interval


def get_bgt_apr(row: Dict[str, str]) -> tuple[float, str]:
    """Extract BGT APR and LP symbol from row data."""
    for key, value in row.items():
        if key.endswith("_bgt_apr"):
            try:
                apr = float(value)
                symbol = key.replace("_bgt_apr", "")
                return apr, symbol
            except (ValueError, TypeError):
                continue
    return 0.0, ""


def get_token_price(row: Dict[str, str], symbol: str) -> float:
    """Get token price from row data."""
    price_key = f"{symbol}_price_in_bera"
    try:
        return float(row.get(price_key, 0.0))
    except (ValueError, TypeError):
        return 0.0


class BacktestProcessor:
    """Handles backtest execution and state management."""

    def __init__(self, initial_position: float, mode: str, compound_interval: int):
        self.position = initial_position
        self.revenue = 0.0
        self.route_symbol = ""
        self.last_compound_ts = 0
        self.mode = mode
        self.compound_interval = compound_interval
        self.results = []

    def process_row(
        self,
        i: int,
        row: Dict[str, str],
        prev_timestamp: int = None,
        is_last_row: bool = False,
    ) -> None:
        """Process a single CSV row."""
        block_number = int(row["block"])
        timestamp = int(row["timestamp"])
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Initialize on first row
        if i == 0:
            self.route_symbol = get_route_symbol(row, self.mode)
            self.last_compound_ts = timestamp
            logging.info(f"   🎯 Initial routing: {self.route_symbol}")

        bgt_apr, lp_symbol = get_bgt_apr(row)

        # Calculate revenue for this period (skip first row)
        if i > 0 and prev_timestamp:
            time_elapsed = timestamp - prev_timestamp
            years_elapsed = time_elapsed / (365 * 24 * 3600)
            period_revenue = (bgt_apr / 100) * self.position * years_elapsed
            self.revenue += period_revenue

        # Check for compounding
        should_compound_now = (
            should_compound(timestamp, self.last_compound_ts, self.compound_interval)
            or is_last_row
        )

        # Handle compounding
        if should_compound_now and i > 0:
            if self.mode == "BGT":
                # BGT mode: compound revenue directly without multiplying by price
                compound_amount = self.revenue
            else:
                # Normal mode: multiply by liquid BGT price
                liquid_bgt_price = get_token_price(row, self.route_symbol)
                compound_amount = self.revenue * liquid_bgt_price
            
            self.position += compound_amount
            self.revenue = 0.0
            self.last_compound_ts = timestamp

            # Update routing for auto mode
            if self.mode == "auto":
                self.route_symbol = get_route_symbol(row, self.mode)

        # Create result row
        if self.mode == "BGT":
            # BGT mode: no liquid BGT price
            result_row = {
                "block": block_number,
                "date": date_str,
                "route_symbol": self.route_symbol,
                f"{lp_symbol}_bgt_apr": bgt_apr,
                "revenue": self.revenue,
                "position": self.position,
            }
        else:
            # Normal mode: include liquid BGT price
            liquid_bgt_price = get_token_price(row, self.route_symbol)
            result_row = {
                "block": block_number,
                "date": date_str,
                "route_symbol": self.route_symbol,
                f"{lp_symbol}_bgt_apr": bgt_apr,
                "liquid_bgt_price_in_bera": liquid_bgt_price,
                "revenue": self.revenue,
                "position": self.position,
            }

        self.results.append(result_row)


def create_csv_fieldnames(results: List[Dict]) -> List[str]:
    """Create ordered fieldnames for CSV output."""
    if not results:
        return []

    all_fields = set()
    for result in results:
        all_fields.update(result.keys())

    # Define field order
    ordered_fields = ["block", "date", "route_symbol"]

    # Add APR fields
    for field in sorted(all_fields):
        if field.endswith("_bgt_apr"):
            ordered_fields.append(field)

    # Add price fields
    for field in sorted(all_fields):
        if field.endswith("_price_in_bera"):
            ordered_fields.append(field)

    # Add remaining fields
    ordered_fields.extend(["revenue", "position"])

    # Remove duplicates while preserving order
    fieldnames = []
    seen = set()
    for field in ordered_fields:
        if field in all_fields and field not in seen:
            fieldnames.append(field)
            seen.add(field)

    return fieldnames


def run_backtest(
    csv_path: str,
    mode: str = "auto",
    interval: str = "0m",
    initial_position: float = 1000000.0,
    output_dir: str = "data/backtest",
) -> str:
    """Execute backtest on CSV data."""
    logging.info("🚀 Starting backtest execution:")
    logging.info(f"   📂 Input CSV: {csv_path}")
    logging.info(f"   🎯 Mode: {mode}")
    logging.info(f"   ⏰ Compound Interval: {interval}")
    logging.info(f"   💰 Initial Position: ${initial_position:,.2f}")

    compound_seconds = parse_interval(interval)
    logging.info(f"   🔄 Compound Every: {compound_seconds}s")

    # Setup output
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_filename = f"{mode}_{interval}_backtest.csv"
    output_path = output_dir_path / output_filename

    # Read CSV data
    with open(csv_path, "r") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty")

    logging.info(f"   📊 Processing {len(rows)} rows")

    # Process backtest
    processor = BacktestProcessor(initial_position, mode, compound_seconds)

    for i, row in enumerate(rows):
        try:
            prev_timestamp = int(rows[i - 1]["timestamp"]) if i > 0 else None
            is_last_row = i == len(rows) - 1
            processor.process_row(i, row, prev_timestamp, is_last_row)

        except Exception as e:
            logging.warning(f"Error processing row {i}: {e}")
            continue

    # Write results
    if processor.results:
        fieldnames = create_csv_fieldnames(processor.results)
        with open(output_path, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processor.results)

    # Summary
    final_position = (
        processor.results[-1]["position"] if processor.results else initial_position
    )
    total_return = ((final_position - initial_position) / initial_position) * 100

    logging.info("\n✅ Backtest completed!")
    logging.info(f"   📊 Rows processed: {len(processor.results)}")
    logging.info(f"   💰 Initial position: ${initial_position:,.2f}")
    logging.info(f"   💰 Final position: ${final_position:,.2f}")
    logging.info(f"   📈 Total return: {total_return:.2f}%")
    logging.info(f"   💾 Output: {output_path}")

    return str(output_path)


def cli() -> None:
    """Command-line interface for backtest execution."""
    parser = argparse.ArgumentParser(
        description="🚀 Execute backtest on LST token strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto routing with continuous compounding
  python scripts/run_backtest_script.py --csv data/data.csv --mode auto --interval 0m

  # iBGT routing with daily compounding
  python scripts/run_backtest_script.py --csv data/data.csv --mode iBGT --interval 1d

  # Auto routing with hourly compounding
  python scripts/run_backtest_script.py --csv data/data.csv --mode auto --interval 1h

  # BGT mode (no liquid BGT price multiplication) with daily compounding
  python scripts/run_backtest_script.py --csv data/data.csv --mode BGT --interval 1d
""",
    )

    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--mode", default="auto", help="Routing mode: 'auto', 'BGT', or specific symbol"
    )
    parser.add_argument(
        "--interval",
        default="0m",
        help="Compound interval: '0m', '30m', '1h', '12h', '1d'",
    )
    parser.add_argument(
        "--position", type=float, default=100.0, help="Initial position size"
    )
    parser.add_argument("--output", default="data/backtest", help="Output directory")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    try:
        if not Path(args.csv).exists():
            raise FileNotFoundError(f"Input CSV not found: {args.csv}")

        output_path = run_backtest(
            csv_path=args.csv,
            mode=args.mode,
            interval=args.interval,
            initial_position=args.position,
            output_dir=args.output,
        )

        logging.info(f"\n🎉 Backtest results saved to: {output_path}")

    except Exception as e:
        logging.error(f"❌ Backtest failed: {e}")
        exit(1)


if __name__ == "__main__":
    cli()
