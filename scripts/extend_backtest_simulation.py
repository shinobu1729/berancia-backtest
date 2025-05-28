#!/usr/bin/env python3
"""
Backtest Extension Simulation Script
===================================

Extends existing backtest data to create 1-year simulations using average APR values.
Generates extended_backtest.csv files by projecting historical performance forward.
"""

import argparse
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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


def should_compound(current_ts: int, last_compound_ts: int, interval: int) -> bool:
    """Check if compounding should occur based on time elapsed."""
    if interval == 0:
        return True
    return (current_ts - last_compound_ts) >= interval


def run_extension_simulation(
    initial_position: float,
    initial_revenue: float,
    avg_liquid_bgt_apr: float,
    compound_interval: int,
    start_date: datetime,
    avg_interval_seconds: int,
    points_needed: int,
) -> List[Dict]:
    """Run extension simulation using the same logic as run_backtest_script.py."""

    position = initial_position
    revenue = initial_revenue
    last_compound_ts = int(start_date.timestamp())
    results = []

    current_date = start_date

    for i in range(points_needed):
        # Update datetime
        current_date += timedelta(seconds=avg_interval_seconds)
        timestamp = int(current_date.timestamp())
        date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")

        # Initialize last_compound_ts on first row
        if i == 0:
            last_compound_ts = timestamp

        # Calculate revenue for this period (skip first row)
        if i > 0:
            time_elapsed = avg_interval_seconds
            years_elapsed = time_elapsed / (365 * 24 * 3600)
            period_revenue = (avg_liquid_bgt_apr / 100) * position * years_elapsed
            revenue += period_revenue

        # Check for compounding
        should_compound_now = (
            should_compound(timestamp, last_compound_ts, compound_interval)
            or i == points_needed - 1  # Force compound on last row
        )

        # Handle compounding (same logic as run_backtest_script.py)
        if should_compound_now and i > 0:
            # Since avg_liquid_bgt_apr already includes price, compound_amount = revenue
            compound_amount = revenue
            position += compound_amount
            revenue = 0.0
            last_compound_ts = timestamp

        # Create result row
        result_row = {
            "block": "simulate",
            "date": date_str,
            "revenue": revenue,
            "position": position,
        }

        results.append(result_row)

    return results


class BacktestExtender:
    """Extends backtest data to simulate full year performance."""

    def __init__(self):
        self.avg_interval_seconds = 197  # ~3.17 minutes average from analysis

    def parse_interval_from_filename(self, filename: str) -> str:
        """Extract interval pattern from filename."""
        stem = Path(filename).stem
        if "_0m_" in stem:
            return "0m"
        elif "_1d_" in stem:
            return "1d"
        else:
            # Extract interval between strategy and 'backtest'
            parts = stem.split("_")
            if len(parts) >= 3 and parts[-1] == "backtest":
                return "_".join(parts[1:-1])
            return "unknown"

    def calculate_average_apr(self, df: pd.DataFrame) -> float:
        """Calculate average liquid BGT APR from backtest data."""
        apr_col = "KODI WBERA-iBGT_bgt_apr"
        price_col = "liquid_bgt_price_in_bera"

        if apr_col not in df.columns or price_col not in df.columns:
            logging.warning("Missing required columns for APR calculation")
            return 0.0

        # Calculate liquid BGT APR: apr * price
        liquid_bgt_apr = df[apr_col] * df[price_col]
        avg_apr = liquid_bgt_apr.mean()

        logging.info(f"Calculated average liquid BGT APR: {avg_apr:.2f}%")
        return avg_apr

    def extend_to_one_year(
        self, df: pd.DataFrame, avg_liquid_bgt_apr: float, interval: str
    ) -> pd.DataFrame:
        """Extend backtest data to simulate 1 year total from first data point."""

        if len(df) == 0:
            raise ValueError("Empty dataframe provided")

        # Get first and last row for parameters
        first_row = df.iloc[0].copy()
        last_row = df.iloc[-1].copy()

        first_date = pd.to_datetime(first_row["date"])
        last_date = pd.to_datetime(last_row["date"])

        # Calculate how much time has already passed
        time_passed = (last_date - first_date).total_seconds()
        total_seconds_in_year = 365 * 24 * 3600

        # Calculate remaining time to reach 1 year from first data
        remaining_seconds = total_seconds_in_year - time_passed

        if remaining_seconds <= 0:
            logging.info("Data already spans more than 1 year, no extension needed")
            return df

        points_needed = int(remaining_seconds / self.avg_interval_seconds)

        logging.info(
            f"Extending from {len(df)} rows to simulate 1 year total "
            f"(adding {points_needed} simulation points)"
        )

        # Parse interval for compounding
        compound_interval_seconds = parse_interval(interval)

        # Extract starting parameters from last row
        initial_position = last_row["position"]
        initial_revenue = last_row["revenue"]

        # Run simulation
        extended_rows = run_extension_simulation(
            initial_position=initial_position,
            initial_revenue=initial_revenue,
            avg_liquid_bgt_apr=avg_liquid_bgt_apr,
            compound_interval=compound_interval_seconds,
            start_date=last_date,
            avg_interval_seconds=self.avg_interval_seconds,
            points_needed=points_needed,
        )

        # Combine original and extended data
        extended_rows_df = pd.DataFrame(extended_rows)
        extended_df = pd.concat([df, extended_rows_df], ignore_index=True)

        logging.info(f"Extended data: {len(df)} -> {len(extended_df)} rows")
        logging.info(f"Final position: {extended_df['position'].iloc[-1]:.2f}")
        logging.info(f"Final revenue: {extended_df['revenue'].iloc[-1]:.2f}")

        return extended_df

    def process_backtest_file(self, file_path: Path, output_dir: str) -> str:
        """Process a single backtest file and create extended version."""

        try:
            # Load original data
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df["date"])

            logging.info(f"Processing {file_path.name}: {len(df)} rows")

            # Extract interval and calculate average APR
            interval = self.parse_interval_from_filename(file_path.name)
            avg_liquid_bgt_apr = self.calculate_average_apr(df)

            logging.info(
                f"Strategy: {file_path.stem}, Interval: {interval}, "
                f"Avg Liquid BGT APR: {avg_liquid_bgt_apr:.2f}%"
            )

            # Extend to 1 year
            extended_df = self.extend_to_one_year(df, avg_liquid_bgt_apr, interval)

            # Save extended data
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            output_filename = f"{file_path.stem}_extended_backtest.csv"
            output_file = output_path / output_filename

            extended_df.to_csv(output_file, index=False)

            logging.info(f"Saved extended backtest: {output_file}")
            return str(output_file)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            raise

    def extend_all_backtests(
        self, backtest_dir: str = "data/backtest", output_dir: str = "data/extend"
    ) -> List[str]:
        """Process all backtest files and create 1-year simulations."""

        backtest_path = Path(backtest_dir)
        if not backtest_path.exists():
            raise FileNotFoundError(f"Backtest directory not found: {backtest_dir}")

        # Find all backtest CSV files
        csv_files = list(backtest_path.glob("*_backtest.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No backtest CSV files found in {backtest_dir}")

        logging.info(f"Found {len(csv_files)} backtest files to extend")

        output_files = []

        for file_path in sorted(csv_files):
            try:
                output_file = self.process_backtest_file(file_path, output_dir)
                output_files.append(output_file)
            except Exception as e:
                logging.warning(f"Skipping {file_path}: {e}")
                continue

        return output_files

    def create_extended_roi_chart(
        self, output_files: List[str], plots_dir: str = "plots"
    ) -> str:
        """Create ROI chart from extended backtest files."""

        # Create plots directory
        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Setup the plot
        plt.figure(figsize=(12, 8))
        plt.style.use("default")

        logging.info(f"Creating extended ROI chart from {len(output_files)} files")

        # Load and plot each extended file
        initial_position = 100.0
        auto_strategy_data = None
        other_strategies = []

        for file_path in output_files:
            try:
                df = pd.read_csv(file_path)
                df["date"] = pd.to_datetime(df["date"])

                # Extract strategy name from filename
                filename = Path(file_path).name
                strategy_name = filename.replace("_extended_backtest.csv", "")

                # Extract base strategy and interval from filename
                base_strategy, interval = self._parse_strategy_filename(strategy_name)

                # Get color and styling from config
                color, label = self._get_strategy_style(base_strategy, interval)

                # Calculate ROI based on strategy type
                # All strategies now use position for ROI calculation
                roi = (df["position"] / initial_position) - 1

                # Store auto strategy separately for emphasis
                if strategy_name.startswith("auto_0m"):
                    auto_strategy_data = (df["date"], roi, color, label)
                else:
                    other_strategies.append((df["date"], roi, color, label))

                logging.info(
                    f"Loaded {strategy_name}: {len(df)} rows, Final ROI: {roi.iloc[-1] * 100:.1f}%"
                )

            except Exception as e:
                logging.warning(f"Skipping {file_path}: {e}")
                continue

        # Plot other strategies first (background)
        for x_axis, roi, color, label in other_strategies:
            plt.plot(
                x_axis,
                roi * 100,
                color=color,
                label=label,
                linewidth=1.5,
                alpha=0.8,
            )

        # Plot auto strategy last (foreground) with emphasis
        if auto_strategy_data:
            x_axis, roi, color, label = auto_strategy_data
            plt.plot(
                x_axis,
                roi * 100,
                color=color,
                label=label,
                linewidth=3.0,
                alpha=1.0,
            )

        # Format chart
        self._format_extended_roi_chart()

        # Save chart
        output_file = plots_path / "extended_backtest_roi_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Created extended ROI chart: {output_file}")
        return str(output_file)

    def _parse_strategy_filename(self, strategy_name: str) -> tuple:
        """Parse strategy filename to extract base strategy and interval.

        Args:
            strategy_name: Strategy name from filename (e.g., 'auto_0m', 'LBGT_1d')

        Returns:
            Tuple of (base_strategy, interval)
        """
        parts = strategy_name.split("_")
        if len(parts) >= 2:
            base_strategy = parts[0]
            interval = parts[1]
            return base_strategy, interval
        else:
            return strategy_name, "unknown"

    def _get_strategy_style(self, base_strategy: str, interval: str) -> tuple:
        """Get color and label for strategy from config.

        Args:
            base_strategy: Base strategy name (e.g., 'auto', 'LBGT')
            interval: Interval string (e.g., '0m', '1d')

        Returns:
            Tuple of (color, label)
        """
        if hasattr(config, "CHART_COLORS") and base_strategy in config.CHART_COLORS:
            color_config = config.CHART_COLORS[base_strategy]
            label = color_config["label"] + f" ({interval})"
            return color_config["color"], label
        else:
            # Fallback if config not available
            return "gray", f"{base_strategy} ({interval})"

    def _format_extended_roi_chart(self) -> None:
        """Apply formatting to the extended ROI chart."""
        # Title and labels
        plt.title(
            "Extended 1-Year ROI Simulation Comparison",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("ROI (%)", fontsize=12)

        # Format y-axis for percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

        # Format x-axis dates
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.figure.autofmt_xdate()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", framealpha=0.9)

        # Tight layout
        plt.tight_layout()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def cli() -> None:
    """Command-line interface for backtest extension."""
    parser = argparse.ArgumentParser(
        description="📈 Extend backtest data to create 1-year performance simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extend all backtest files to 1 year
  python scripts/extend_backtest_simulation.py

  # Specify custom directories
  python scripts/extend_backtest_simulation.py --backtest-dir data/backtest --output data/extended

  # With verbose logging
  python scripts/extend_backtest_simulation.py --verbose
""",
    )

    parser.add_argument(
        "--backtest-dir",
        default="data/backtest",
        help="Directory containing original backtest CSV files",
    )

    parser.add_argument(
        "--output",
        default="data/extend",
        help="Output directory for extended backtest files",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Create extender and process files
        logging.info("🚀 Starting backtest extension to 1-year simulation")
        logging.info(f"   📂 Input directory: {args.backtest_dir}")
        logging.info(f"   📊 Output directory: {args.output}")

        extender = BacktestExtender()
        output_files = extender.extend_all_backtests(args.backtest_dir, args.output)

        if output_files:
            logging.info(f"\n✅ Created {len(output_files)} extended backtest files:")
            for file_path in output_files:
                logging.info(f"   📈 {file_path}")

            # Create ROI chart from extended data
            try:
                chart_file = extender.create_extended_roi_chart(output_files)
                logging.info(f"\n📊 Created extended ROI chart: {chart_file}")
            except Exception as e:
                logging.error(f"Failed to create ROI chart: {e}")
        else:
            logging.warning("No extended files were created")

    except Exception as e:
        logging.error(f"❌ Extension failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
