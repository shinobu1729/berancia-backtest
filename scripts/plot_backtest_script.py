#!/usr/bin/env python3
"""
Backtest Visualization Script
============================

Creates ROI performance charts from backtest CSV files.
Supports multiple strategies with configurable colors and proper date formatting.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config


class BacktestDataLoader:
    """Handles loading and processing of backtest CSV files."""

    @staticmethod
    def find_backtest_files(backtest_dir: str) -> List[Path]:
        """Find all backtest CSV files in the directory."""
        backtest_path = Path(backtest_dir)
        if not backtest_path.exists():
            raise FileNotFoundError(f"Backtest directory not found: {backtest_dir}")

        csv_files = list(backtest_path.glob("*_backtest.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No backtest CSV files found in {backtest_dir}")

        return sorted(csv_files)

    @staticmethod
    def parse_filename(file_path: Path) -> Tuple[str, str]:
        """Parse strategy and interval from filename."""
        # Format: {strategy}_{interval}_backtest.csv
        stem = file_path.stem  # Remove .csv
        parts = stem.split("_")

        if len(parts) >= 3 and parts[-1] == "backtest":
            strategy = parts[0]
            interval = "_".join(parts[1:-1])  # Handle complex intervals
            return strategy, interval

        # Fallback
        return file_path.stem, "unknown"

    @staticmethod
    def load_backtest_data(file_path: Path) -> pd.DataFrame:
        """Load backtest CSV data into DataFrame."""
        try:
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                logging.debug(
                    f"Loaded {file_path.name}: {len(df)} rows, {df['date'].iloc[0]} to {df['date'].iloc[-1]}"
                )

            # Ensure position column exists
            if "position" not in df.columns:
                raise ValueError(f"No 'position' column found in {file_path}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def load_all_strategies(self, backtest_dir: str) -> Dict[str, pd.DataFrame]:
        """Load all backtest strategies from directory."""
        csv_files = self.find_backtest_files(backtest_dir)
        logging.info(f"Found {len(csv_files)} backtest files")

        all_strategies = {}

        for file_path in csv_files:
            strategy, interval = self.parse_filename(file_path)

            # Create unique key with strategy and interval
            strategy_key = (
                f"{strategy}_{interval}"
                if strategy != "auto"
                else f"berancia_{interval}"
            )

            # Load data
            try:
                df = self.load_backtest_data(file_path)
                all_strategies[strategy_key] = df
                logging.info(f"Loaded {strategy_key}: {len(df)} rows")
            except Exception as e:
                logging.warning(f"Skipping {file_path}: {e}")
                continue

        return all_strategies


class ChartStyler:
    """Handles chart styling and color configuration."""

    @staticmethod
    def get_strategy_color_and_label(strategy_key: str) -> Tuple[str, str]:
        """Get color and label for strategy from config."""
        # Extract base strategy from key (e.g., "iBGT_1d" -> "iBGT")
        if strategy_key.startswith("berancia_"):
            base_strategy = "auto"
            interval = strategy_key.replace("berancia_", "")
            label_suffix = f" ({interval})"
        else:
            parts = strategy_key.split("_")
            base_strategy = parts[0]
            interval = "_".join(parts[1:]) if len(parts) > 1 else ""
            label_suffix = f" ({interval})" if interval else ""

        if base_strategy in config.CHART_COLORS:
            color_config = config.CHART_COLORS[base_strategy]
            label = color_config["label"] + label_suffix
            return color_config["color"], label

        # Fallback for unknown strategies
        return "gray", strategy_key

    @staticmethod
    def configure_date_axis(ax, first_df: pd.DataFrame) -> None:
        """Configure X-axis date formatting based on data span."""
        if "date" not in first_df.columns or len(first_df) == 0:
            logging.info("No date column found, using default x-axis formatting")
            return

        # Calculate appropriate interval based on data length
        data_span_days = (first_df["date"].iloc[-1] - first_df["date"].iloc[0]).days
        logging.info(
            f"Data spans {data_span_days} days from {first_df['date'].iloc[0]} to {first_df['date'].iloc[-1]}"
        )

        # Set date formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        # Set appropriate locator based on data span
        if data_span_days > 56:
            logging.info(f"Using weekly interval for {data_span_days} day span")
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
        elif data_span_days > 28:
            logging.info(f"Using 7-day interval for {data_span_days} day span")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        elif data_span_days > 14:
            logging.info(f"Using 3-day interval for {data_span_days} day span")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        elif data_span_days > 7:
            logging.info(f"Using 2-day interval for {data_span_days} day span")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        else:
            logging.info(f"Using daily interval for {data_span_days} day span")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

        # Apply formatting
        ax.figure.autofmt_xdate()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        logging.info("Date formatting applied successfully")


class ROIChartCreator:
    """Creates ROI performance charts from backtest data."""

    def __init__(self, chart_styler: ChartStyler):
        self.chart_styler = chart_styler

    def create_roi_chart(
        self, data_dict: Dict[str, pd.DataFrame], output_dir: str = "plots"
    ) -> str:
        """Create ROI performance chart for all strategies."""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup the plot
        plt.figure(figsize=config.CHART_FIGURE_SIZE)
        plt.style.use("default")

        # Get initial position from first strategy (should be same for all)
        first_df = next(iter(data_dict.values()))
        initial_position = (
            first_df["position"].iloc[0]
            if len(first_df) > 0
            else config.CHART_DEFAULT_INITIAL_POSITION
        )

        logging.info(
            f"Creating ROI chart with {len(data_dict)} strategies, initial position: {initial_position}"
        )

        # Plot each strategy
        self._plot_strategies(data_dict, initial_position)

        # Format chart
        self._format_chart(first_df)

        # Save chart
        output_file = output_path / config.CHART_OUTPUT_FILENAME
        plt.savefig(output_file, dpi=config.CHART_DPI, bbox_inches="tight")
        plt.close()

        return str(output_file)

    def _plot_strategies(
        self, data_dict: Dict[str, pd.DataFrame], initial_position: float
    ) -> None:
        """Plot all strategies on the chart."""
        for strategy, df in data_dict.items():
            color, label = self.chart_styler.get_strategy_color_and_label(strategy)

            # Calculate ROI: (position / initial_position) - 1
            roi = (df["position"] / initial_position) - 1

            # Use date if available, otherwise use index
            if "date" in df.columns:
                x_axis = df["date"]
                plt.plot(
                    x_axis,
                    roi * 100,
                    color=color,
                    label=label,
                    linewidth=config.CHART_LINE_WIDTH,
                    alpha=config.CHART_LINE_ALPHA,
                )
                logging.debug(
                    f"Plotted {strategy} with dates from {x_axis.iloc[0]} to {x_axis.iloc[-1]}"
                )
            else:
                x_axis = range(len(df))
                plt.plot(
                    x_axis,
                    roi * 100,
                    color=color,
                    label=label,
                    linewidth=config.CHART_LINE_WIDTH,
                    alpha=config.CHART_LINE_ALPHA,
                )
                logging.debug(f"Plotted {strategy} with index range 0 to {len(df)}")

    def _format_chart(self, first_df: pd.DataFrame) -> None:
        """Apply formatting to the chart."""
        # Title and labels
        plt.title(
            config.CHART_TITLE,
            fontsize=config.CHART_TITLE_FONTSIZE,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Date", fontsize=config.CHART_AXIS_FONTSIZE)
        plt.ylabel("ROI (%)", fontsize=config.CHART_AXIS_FONTSIZE)

        # Format y-axis for percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

        # Format x-axis dates
        ax = plt.gca()
        logging.info("Applying date formatting to X-axis...")
        self.chart_styler.configure_date_axis(ax, first_df)

        # Grid and legend
        plt.grid(True, alpha=config.CHART_GRID_ALPHA)
        plt.legend(
            loc=config.CHART_LEGEND_LOCATION, framealpha=config.CHART_LEGEND_ALPHA
        )

        # Tight layout
        plt.tight_layout()

    def create_apr_yield_chart(
        self, data_dict: Dict[str, pd.DataFrame], output_dir: str = "plots"
    ) -> str:
        """Create APR yield chart using original data for individual tokens and backtest data for Berancia."""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup the plot
        plt.figure(figsize=config.CHART_FIGURE_SIZE)
        plt.style.use("default")

        logging.info(f"Creating APR yield chart with {len(data_dict)} strategies")

        # Load original data for individual token calculations
        self._plot_apr_yield_from_original_data(data_dict)

        # Get first df for date formatting
        first_df = next(iter(data_dict.values()))

        # Format chart
        self._format_apr_yield_chart(first_df)

        # Save chart
        output_file = output_path / "backtest_apr_yield_comparison.png"
        plt.savefig(output_file, dpi=config.CHART_DPI, bbox_inches="tight")
        plt.close()

        return str(output_file)

    def _plot_apr_yield_from_original_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Plot APR yield using original data.csv for individual tokens and backtest data for Berancia."""
        try:
            # Load original data
            original_data_path = Path(__file__).parent.parent / "data" / "data.csv"
            original_df = pd.read_csv(original_data_path)
            # Convert timestamp to datetime in JST (UTC+9) to match backtest data
            original_df['timestamp'] = pd.to_datetime(original_df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
            
            # Find auto strategy for Berancia
            auto_strategy = None
            for strategy, df in data_dict.items():
                if strategy.startswith("berancia_") or "auto_" in strategy:
                    auto_strategy = (strategy, df)
                    break
            
            # Add routing switch indicators if we have auto strategy
            if auto_strategy:
                strategy, df = auto_strategy
                self._add_routing_switch_lines(df)
            
            # Plot individual token strategies from original data
            for strategy, df in data_dict.items():
                if not (strategy.startswith("berancia_") or "auto_" in strategy):
                    self._plot_individual_token_from_original_data(strategy, original_df)
            
            # Plot yBGT from timestamp 1743839065 onwards
            self._plot_ybgt_from_original_data(original_df)
            
            # Plot Berancia from backtest data
            if auto_strategy:
                strategy, df = auto_strategy
                self._plot_berancia_from_backtest_data(strategy, df)
                
        except Exception as e:
            logging.error(f"Error plotting APR yield from original data: {e}")
            # Fallback to old method
            self._plot_apr_yield_strategies(data_dict)

    def _plot_individual_token_from_original_data(self, strategy: str, original_df: pd.DataFrame) -> None:
        """Plot individual token APR × price from original data."""
        try:
            # Extract token name from strategy (e.g., "LBGT_1d" -> "LBGT")
            token_name = strategy.split('_')[0]
            
            apr_col = "KODI WBERA-iBGT_bgt_apr"
            price_col = f"{token_name}_price_in_bera"
            
            if apr_col in original_df.columns and price_col in original_df.columns:
                # Calculate APR × price
                apr_yield = original_df[apr_col] * original_df[price_col]
                
                color, label = self.chart_styler.get_strategy_color_and_label(strategy)
                
                plt.plot(
                    original_df['timestamp'],
                    apr_yield,
                    color=color,
                    label=label,
                    linewidth=1.5,
                    alpha=1.0
                )
                
                logging.info(f"Plotted {strategy} from original data with {len(original_df)} points")
            else:
                logging.warning(f"Missing columns for {strategy}: {apr_col} or {price_col}")
                
        except Exception as e:
            logging.warning(f"Could not plot {strategy} from original data: {e}")

    def _plot_ybgt_from_original_data(self, original_df: pd.DataFrame) -> None:
        """Plot yBGT APR × price from timestamp 1743839065 onwards."""
        try:
            # Filter for yBGT data from timestamp 1743839065 onwards (convert to JST)
            target_timestamp = pd.to_datetime(1743839065, unit='s', utc=True).tz_convert('Asia/Tokyo').tz_localize(None)
            ybgt_mask = original_df['timestamp'] >= target_timestamp
            ybgt_data = original_df[ybgt_mask]
            
            apr_col = "KODI WBERA-iBGT_bgt_apr"
            price_col = "yBGT_price_in_bera"
            
            if len(ybgt_data) > 0 and apr_col in ybgt_data.columns and price_col in ybgt_data.columns:
                # Calculate yBGT APR × price
                ybgt_apr_yield = ybgt_data[apr_col] * ybgt_data[price_col]
                
                plt.plot(
                    ybgt_data['timestamp'],
                    ybgt_apr_yield,
                    color='blue',
                    label='yBGT APR × Price (from 4/10)',
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle='--'
                )
                
                logging.info(f"Plotted yBGT from original data with {len(ybgt_data)} points from timestamp 1743839065")
            else:
                logging.warning("No yBGT data found from timestamp 1743839065 onwards")
                
        except Exception as e:
            logging.warning(f"Could not plot yBGT from original data: {e}")

    def _plot_berancia_from_backtest_data(self, strategy: str, df: pd.DataFrame) -> None:
        """Plot Berancia APR × price from backtest data."""
        try:
            apr_col = "KODI WBERA-iBGT_bgt_apr"
            price_col = "liquid_bgt_price_in_bera"
            
            if apr_col in df.columns and price_col in df.columns:
                # Calculate Berancia APR × price
                apr_yield = df[apr_col] * df[price_col]
                
                color, label = self.chart_styler.get_strategy_color_and_label(strategy)
                
                if "date" in df.columns:
                    x_axis = pd.to_datetime(df["date"])
                    plt.plot(
                        x_axis,
                        apr_yield,
                        color=color,
                        label=label,
                        linewidth=2.0,
                        alpha=1.0
                    )
                    
                    logging.info(f"Plotted Berancia from backtest data with {len(df)} points")
                else:
                    logging.warning("No date column found in Berancia backtest data")
            else:
                logging.warning(f"Missing columns for Berancia: {apr_col} or {price_col}")
                
        except Exception as e:
            logging.warning(f"Could not plot Berancia from backtest data: {e}")

    def _plot_apr_yield_strategies(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Plot APR yield strategies on the chart."""
        # Special handling for auto strategy to make it prominent
        auto_strategy = None
        other_strategies = {}

        for strategy, df in data_dict.items():
            if strategy.startswith("berancia_") or "auto_" in strategy:
                auto_strategy = (strategy, df)
            else:
                other_strategies[strategy] = df

        # Add routing switch indicators first (background)
        if auto_strategy:
            strategy, df = auto_strategy
            self._add_routing_switch_lines(df)

        # Plot yBGT APR x price curve from April 10th onwards
        if auto_strategy:
            strategy, df = auto_strategy
            self._plot_ybgt_apr_yield_curve(df)

        # Plot other strategies
        for strategy, df in other_strategies.items():
            self._plot_single_apr_yield_strategy(strategy, df, linewidth=1.5, alpha=1)

        # Plot auto strategy last (foreground) with emphasis
        if auto_strategy:
            strategy, df = auto_strategy
            self._plot_single_apr_yield_strategy(strategy, df, linewidth=2.0, alpha=1.0)

    def _plot_single_apr_yield_strategy(
        self, strategy: str, df: pd.DataFrame, linewidth: float, alpha: float
    ) -> None:
        """Plot a single APR yield strategy."""
        color, label = self.chart_styler.get_strategy_color_and_label(strategy)

        # Check for required columns
        apr_col = "KODI WBERA-iBGT_bgt_apr"
        price_col = "liquid_bgt_price_in_bera"

        if apr_col not in df.columns or price_col not in df.columns:
            logging.warning(f"Missing required columns in {strategy}, skipping")
            return

        # Calculate y = apr * price
        apr_yield = df[apr_col] * df[price_col]

        # Use date if available, otherwise use index
        if "date" in df.columns:
            x_axis = df["date"]
            plt.plot(
                x_axis,
                apr_yield,
                color=color,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
            )
            logging.debug(
                f"Plotted {strategy} APR yield from {x_axis.iloc[0]} to {x_axis.iloc[-1]}"
            )
        else:
            x_axis = range(len(df))
            plt.plot(
                x_axis,
                apr_yield,
                color=color,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
            )
            logging.debug(
                f"Plotted {strategy} APR yield with index range 0 to {len(df)}"
            )

    def _format_apr_yield_chart(self, first_df: pd.DataFrame) -> None:
        """Apply formatting to the APR yield chart."""
        # Title and labels
        plt.title(
            "BGT APR × Liquid BGT Price Comparison",
            fontsize=config.CHART_TITLE_FONTSIZE,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Date", fontsize=config.CHART_AXIS_FONTSIZE)
        plt.ylabel("APR × Price", fontsize=config.CHART_AXIS_FONTSIZE)

        # Format x-axis dates
        ax = plt.gca()
        logging.info("Applying date formatting to X-axis...")
        self.chart_styler.configure_date_axis(ax, first_df)

        # Grid and legend
        plt.grid(True, alpha=config.CHART_GRID_ALPHA)
        plt.legend(
            loc=config.CHART_LEGEND_LOCATION, framealpha=config.CHART_LEGEND_ALPHA
        )

        # Tight layout
        plt.tight_layout()

    def _plot_ybgt_apr_yield_curve(self, auto_df: pd.DataFrame) -> None:
        """Plot yBGT APR x price curve from April 10th onwards."""
        try:
            # Convert date column to datetime if it's not already
            auto_df_copy = auto_df.copy()
            if 'date' in auto_df_copy.columns:
                auto_df_copy['date'] = pd.to_datetime(auto_df_copy['date'])
                
                # Filter for data from April 10th onwards where route_symbol is yBGT
                april_10 = pd.to_datetime('2025-04-10')
                ybgt_mask = (auto_df_copy['date'] >= april_10) & (auto_df_copy['route_symbol'] == 'yBGT')
                ybgt_data = auto_df_copy[ybgt_mask]
                
                if len(ybgt_data) > 0:
                    # Check for required columns
                    apr_col = "KODI WBERA-iBGT_bgt_apr"
                    price_col = "liquid_bgt_price_in_bera"
                    
                    if apr_col in ybgt_data.columns and price_col in ybgt_data.columns:
                        # Calculate yBGT APR x price
                        ybgt_apr_yield = ybgt_data[apr_col] * ybgt_data[price_col]
                        
                        # Plot yBGT curve with distinctive styling
                        plt.plot(
                            ybgt_data['date'],
                            ybgt_apr_yield,
                            color='purple',
                            label='yBGT APR × Price (from 4/10)',
                            linewidth=1.5,
                            alpha=0.8,
                            linestyle='--'
                        )
                        
                        logging.info(f"Added yBGT APR x price curve with {len(ybgt_data)} data points from April 10th")
                    else:
                        logging.warning("Missing required APR or price columns for yBGT curve")
                else:
                    logging.info("No yBGT data found from April 10th onwards")
            else:
                logging.warning("No date column found in auto strategy data")
                
        except Exception as e:
            logging.warning(f"Could not add yBGT APR x price curve: {e}")

    def _add_routing_switch_lines(self, auto_df: pd.DataFrame) -> None:
        """Add vertical lines to indicate routing switches in auto strategy."""
        try:
            # Find routing switches
            auto_df_copy = auto_df.copy()
            auto_df_copy['prev_route'] = auto_df_copy['route_symbol'].shift(1)
            
            # Find where routing symbol changes
            switch_mask = auto_df_copy['route_symbol'] != auto_df_copy['prev_route']
            routing_switches = auto_df_copy[switch_mask].dropna()
            
            if len(routing_switches) == 0:
                logging.info("No routing switches found in auto strategy")
                return
            
            # Add vertical lines at all switch points
            for _, row in routing_switches.iterrows():
                switch_date = row['date']
                
                # Add vertical line in black color with lower opacity
                plt.axvline(
                    x=switch_date,
                    color='black',
                    linestyle='--',
                    alpha=0.3,
                    linewidth=0.5,
                    zorder=0  # Put lines in background
                )
            
            # Add legend entry for routing switches
            if len(routing_switches) > 0:
                # Create a dummy line for legend
                plt.plot([], [], color='black', linestyle='--', alpha=0.3, linewidth=0.5, 
                        label=f'Auto Routing Switches ({len(routing_switches)} total)')
            
            logging.info(f"Added {len(routing_switches)} routing switch indicators")
            
        except Exception as e:
            logging.warning(f"Could not add routing switch indicators: {e}")


class BacktestVisualizer:
    """Main class for backtest visualization."""

    def __init__(self):
        self.data_loader = BacktestDataLoader()
        self.chart_styler = ChartStyler()
        self.chart_creator = ROIChartCreator(self.chart_styler)

    def plot_backtest_results(
        self, backtest_dir: str = "data/backtest", output_dir: str = "plots"
    ) -> List[str]:
        """Create ROI and APR yield charts for all backtest strategies."""

        # Load all strategies
        all_strategies = self.data_loader.load_all_strategies(backtest_dir)

        # Create charts if data exists
        output_files = []

        if len(all_strategies) > 0:
            try:
                # Create ROI chart
                roi_output_file = self.chart_creator.create_roi_chart(
                    all_strategies, output_dir
                )
                output_files.append(roi_output_file)
                logging.info(f"Created ROI chart: {roi_output_file}")

                # Create APR yield chart
                apr_output_file = self.chart_creator.create_apr_yield_chart(
                    all_strategies, output_dir
                )
                output_files.append(apr_output_file)
                logging.info(f"Created APR yield chart: {apr_output_file}")

            except Exception as e:
                logging.error(f"Error creating charts: {e}")
        else:
            logging.warning("No valid data found for charting")

        return output_files


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def cli() -> None:
    """Command-line interface for backtest plotting."""
    parser = argparse.ArgumentParser(
        description="📊 Create ROI performance charts from backtest CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all backtest files in default directory
  python scripts/plot_backtest_script.py

  # Plot specific directory
  python scripts/plot_backtest_script.py --backtest-dir data/backtest --output plots

  # With verbose logging
  python scripts/plot_backtest_script.py --verbose
""",
    )

    parser.add_argument(
        "--backtest-dir",
        default=config.DEFAULT_BACKTEST_DIR,
        help="Directory containing backtest CSV files",
    )

    parser.add_argument(
        "--output", default=config.DEFAULT_PLOTS_DIR, help="Output directory for charts"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Create visualizer and generate charts
        logging.info("🚀 Starting backtest visualization")
        logging.info(f"   📂 Backtest directory: {args.backtest_dir}")
        logging.info(f"   📊 Output directory: {args.output}")

        visualizer = BacktestVisualizer()
        output_files = visualizer.plot_backtest_results(args.backtest_dir, args.output)

        if output_files:
            logging.info(f"\n✅ Created {len(output_files)} charts:")
            for file_path in output_files:
                logging.info(f"   📈 {file_path}")
        else:
            logging.warning("No charts were created")

    except Exception as e:
        logging.error(f"❌ Plotting failed: {e}")
        exit(1)


if __name__ == "__main__":
    cli()
