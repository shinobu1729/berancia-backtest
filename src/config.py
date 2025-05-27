"""
Configuration file for Berachain DeFi analytics platform.

This module contains all configuration settings for:
- Blockchain connection and API endpoints
- Token addresses and contract configurations
- Chart styling and visualization settings
- Processing parameters and performance tuning
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, try to load from environment
    pass

# =============================================================================
# BLOCKCHAIN CONFIGURATION
# =============================================================================

# Alchemy API configuration (loaded from environment)
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "")
NETWORK = "berachain-mainnet"

# External service endpoints (loaded from environment)
GOLDSKY_ENDPOINT = os.getenv("GOLDSKY_ENDPOINT", "")

# Alchemy RPC URL template
ALCHEMY_RPC_URL_TEMPLATE = "https://{network}.g.alchemy.com/v2/{api_key}"

# Environment variable validation
if not ALCHEMY_API_KEY:
    raise ValueError(
        "ALCHEMY_API_KEY environment variable is required. "
        "Please set it in your .env file or environment."
    )

if not GOLDSKY_ENDPOINT:
    raise ValueError(
        "GOLDSKY_ENDPOINT environment variable is required. "
        "Please set it in your .env file or environment."
    )


# =============================================================================
# TOKEN ADDRESSES
# =============================================================================

# Core token addresses
BERA_ADDRESS = "0x6969696969696969696969696969696969696969"  # BERA token address
HONEY_ADDRESS = "0xfcbd14dc51f0a4d49d5e53c2e0950e0bc26d0dce"  # HONEY token address

# Target RewardVault addresses
REWARD_VAULT_ADDRESSES = [
    "0x3Be1bE98eFAcA8c1Eb786Cbf38234c84B5052EeB",
    # Add other vault addresses as needed
]

# LST (Liquid Staking Token) configurations
LST_TOKENS = [
    {
        "symbol": "iBGT",
        "address": "0xac03caba51e17c86c921e1f6cbfbdc91f8bb2e6b",
        "color": "red",
    },
    {
        "symbol": "LBGT",
        "address": "0xbaadcc2962417c01af99fb2b7c75706b9bd6babe",
        "color": "purple",
    },
    {
        "symbol": "yBGT",
        "address": "0x7e768f47dfdd5dae874aac233f1bc5817137e453",
        "color": "blue",
    },
    {
        "symbol": "stBGT",
        "address": "0x2cec7f1ac87f5345ced3d6c74bbb61bfae231ffb",
        "color": "green",
    },
]


# =============================================================================
# CHART AND VISUALIZATION SETTINGS
# =============================================================================

# Strategy colors for charts
CHART_COLORS = {
    "auto": {"color": "orange", "label": "berancia"},
    "iBGT": {"color": "red", "label": "iBGT"},
    "LBGT": {"color": "purple", "label": "LBGT"},
    "yBGT": {"color": "blue", "label": "yBGT"},
    "stBGT": {"color": "green", "label": "stBGT"},
}

# Chart appearance settings
CHART_FIGURE_SIZE = (12, 8)
CHART_DPI = 300
CHART_TITLE = "LST Strategy ROI Comparison"
CHART_TITLE_FONTSIZE = 16
CHART_AXIS_FONTSIZE = 12
CHART_LINE_WIDTH = 2
CHART_LINE_ALPHA = 0.8
CHART_GRID_ALPHA = 0.3
CHART_LEGEND_LOCATION = "upper left"
CHART_LEGEND_ALPHA = 0.9
CHART_OUTPUT_FILENAME = "backtest_roi_comparison.png"
CHART_DEFAULT_INITIAL_POSITION = 1000.0


# =============================================================================
# DEFAULT DIRECTORIES
# =============================================================================

DEFAULT_BACKTEST_DIR = "data/backtest"
DEFAULT_PLOTS_DIR = "plots"
DEFAULT_ORIGINAL_DATA_DIR = "data/original"
DEFAULT_ANALYTICS_DIR = "data/analytics"


# =============================================================================
# NUMERICAL CALCULATION SETTINGS
# =============================================================================

# Precision constants
PRECISION = 10**18  # Standard precision for token amounts and LP tokens
PRECISION_E36 = 10**36  # Precision for reward rates

# Time constants
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
WEEKS_PER_YEAR = 52


# =============================================================================
# PERFORMANCE AND CACHING SETTINGS
# =============================================================================

# Pool cache settings
POOL_CACHE_REFRESH_BLOCKS = 45000  # Refresh pool addresses from subgraph every N blocks
POOL_CACHE_FILE = "data/.pool_cache.json"  # Pool address cache file path

# Batch processing settings
BATCH_LOOP_COUNT = 100  # Number of blocks to process in a single mega-batch
DEFAULT_BATCH_SIZE = 500  # Default batch size for CSV operations
DEFAULT_BLOCK_INTERVAL = 2000  # Default block interval for data collection


# =============================================================================
# DEBUG AND LOGGING SETTINGS
# =============================================================================

# Debug settings
DEBUG = False
VERBOSE_LOGGING = False
LOG_FORMAT = "[%(levelname)s] %(message)s"

# Request timeout settings
RPC_TIMEOUT = 30
SUBGRAPH_TIMEOUT = 30
MEGA_BATCH_TIMEOUT = 60


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

# Default backtest parameters
DEFAULT_INITIAL_POSITION = 1000.0  # Starting position in BERA
DEFAULT_COMPOUND_INTERVALS = ["0m", "24h", "168h"]  # Continuous, daily, weekly

# Compound interval mappings (in hours)
COMPOUND_INTERVALS = {
    "0m": 0,  # Continuous (every block)
    "24h": 24,  # Daily
    "168h": 168,  # Weekly
}

# Strategy options
AVAILABLE_STRATEGIES = ["auto"] + [token["symbol"] for token in LST_TOKENS]
DEFAULT_STRATEGY = "auto"
