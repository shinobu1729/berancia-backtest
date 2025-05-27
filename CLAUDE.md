# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **DeFi analytics platform** for optimizing **Berachain LST (Liquid Staking Token) investment strategies**. The system performs automated backtesting of yield farming strategies that route capital to the highest-priced LST tokens while tracking and compounding LP rewards.

## Architecture

### Core Modules
- **`src/blockchain/`** - Web3 blockchain interaction via Alchemy API
  - `blockchain_reader.py` - Core blockchain data fetching
  - `optimized_batch_collector.py` - Optimized batch data collection
  - `read_reward_vault.py` - Reward vault APR calculations
- **`src/analysis/`** - Financial calculations and LP value analysis
  - `calculate_lp_value.py` - LP token value calculations
- **`src/abi/`** - Smart contract ABIs
  - `BeraRouter.py`, `ERC20.py`, `KodiakIslandWithRouter.py`, etc.
- **`src/utils/`** - Utility functions
  - `vault_utils.py` - Vault-related helper functions

### Data Flow
```
Raw Prices (LST tokens) → Routing Strategy → LP Rewards → Compound Interest → Performance Analytics → Visualization
```

### Key Configuration (`src/config.py`)
- **Target LP**: `0x564f011D557aAd1cA09BFC956Eb8a17C35d490e0`
- **Reward Vault**: `0x3Be1bE98eFAcA8c1Eb786Cbf38234c84B5052EeB`
- **LST Tokens**: iBGT, LBGT, yBGT, stBGT with their contract addresses
- **Network**: Berachain mainnet via Alchemy API

## Common Commands

### Primary Workflow
```bash
# Step 1: Generate blockchain data
python scripts/query_data.py

# Step 2: Run backtest with routing strategies
# Auto routing (highest price LST)
python scripts/run_backtest_script.py --strategy auto

# Specific token routing
python scripts/run_backtest_script.py --strategy iBGT

# Always calculates 3 compound strategies: continuous, 24h, 168h
```

### Data Management
```bash
# Validate generated data integrity
python scripts/validate_data.py

# Append new data to existing dataset
python scripts/append_data.py
```

### Analysis and Visualization
```bash
# Generate backtest performance plots
python scripts/plot_backtest_script.py
```

### Development Commands
```bash
# Run backtest with verbose logging
python scripts/run_backtest_script.py --strategy auto --verbose

# Custom strategy parameters
python scripts/run_backtest_script.py --strategy LBGT
```

## Data Architecture

### Data Architecture
1. **Data Generation**: `query_data.py` creates base CSV with:
   - `block, timestamp, apr, {symbol}_price_in_bera` (one column per LST token)
   - Saved as `data/data.csv`

2. **Backtest Execution**: `run_backtest_script.py` processes data and outputs:
   - `block, timestamp, apr, {symbol}_price_in_bera, route_symbol, route_price, liquid_bgt_apr, compound_every, compound_24h, compound_168h`
   - Saved to `data/backtest/` directory

### Routing Strategies
- **"auto"**: Automatically routes to LST token with highest price
- **Specific symbol**: Routes to specific token (e.g., "iBGT", "LBGT", "yBGT", "stBGT")

### Compound Strategies (Always Calculated)
- **compound_every**: Continuous compounding (every block interval)
- **compound_24h**: Compounds every 24 hours
- **compound_168h**: Compounds every 168 hours (weekly)

## Development Patterns

- **CLI-first architecture** - All functionality accessible via scripts in `scripts/` directory
- **Configuration-driven** - Token addresses and settings centralized in `src/config.py`
- **Modular blockchain interaction** - Web3 logic separated into dedicated modules
- **Structured data flow** - Original data → Backtest results → Analytics → Visualization
- **Flexible routing** - Support for both automatic and manual token selection strategies

## Key Dependencies
- **web3.py** - Blockchain interaction
- **pandas** - Data analysis and CSV processing
- **matplotlib** - Plotting and visualization
- **Alchemy API** - Berachain mainnet access
- **Python 3.12+** - Runtime environment

## File Structure
```
data/
├── data.csv      # Main dataset: block, timestamp, apr, LST prices
├── backtest/     # Backtest results with routing and compounding strategies
└── .pool_cache.json  # Cached pool addresses for performance optimization
plots/            # Generated visualization outputs
```
