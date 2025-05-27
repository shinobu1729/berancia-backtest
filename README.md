# APR Computer

DeFi analytics platform for optimizing Berachain LST (Liquid Staking Token) investment strategies. The system performs automated backtesting of yield farming strategies that route capital to the highest-priced LST tokens while tracking and compounding LP rewards.

## Quick Start

### Data Generation and Backtesting

```bash
# Generate blockchain data
python scripts/query_data.py

# Run backtest with auto routing (highest price LST)
python scripts/run_backtest_script.py --strategy auto

# Run backtest with specific token routing
python scripts/run_backtest_script.py --strategy iBGT

# Validate generated data
python scripts/validate_data.py

# Generate visualization plots
python scripts/plot_backtest_script.py
```

### Data Management

```bash
# Append new data to existing dataset
python scripts/append_data.py
```

## Development Setup

### Code Formatting

This project uses automated code formatting:

- **Black**: Python code formatter (88 character line length)
- **isort**: Import sorting and organization
- **flake8**: Code linting and style checking
- **pre-commit**: Automatic formatting on git commits

#### Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Manual Formatting

```bash
# Format all Python files
black scripts/ src/

# Sort imports
isort scripts/ src/

# Check code style
flake8 scripts/ src/

# Run all pre-commit hooks manually
pre-commit run --all-files
```

## Architecture

### Core Components
- **scripts/**: CLI tools for data generation, backtesting, and analysis
- **src/blockchain/**: Web3 blockchain interaction and data collection
- **src/analysis/**: Financial calculations and LP value analysis
- **src/abi/**: Smart contract ABIs for Berachain protocols
- **src/config.py**: Centralized configuration
- **src/utils/**: Utility functions for vault operations

### Data Flow
```
Blockchain Data → Query/Generate → Backtest Strategies → Compound Interest → Performance Analytics → Visualization
```

### Key Features
- **Multi-strategy backtesting**: Auto-routing vs specific token strategies
- **Compound interest modeling**: Multiple compounding frequencies (continuous, 24h, 168h)
- **LP reward tracking**: Integration with Berachain reward vaults
- **Performance optimization**: Batch RPC calls and caching
- **Real-time monitoring**: Progress tracking with ETA calculations

### File Structure
```
data/
├── backtest/     # Backtest results with routing and compounding data
├── data.csv      # Main dataset
plots/            # Generated visualization outputs
```
