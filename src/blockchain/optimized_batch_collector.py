"""
Optimized Batch Data Collector
=============================

High-performance blockchain data collection with:
- Pool address caching to reduce subgraph queries
- Batched RPC calls for APR and price data
- Intelligent cache refresh mechanism
- Error handling for individual batch call failures
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from src import config
from src.blockchain.blockchain_reader import AlchemyBlockReader
from src.blockchain.read_reward_vault import RewardVaultAnalyzer
from src.utils.vault_utils import get_all_vault_stake_tokens


class OptimizedBatchCollector:
    """Optimized batch data collector for blockchain metrics."""

    def __init__(self, api_key: str, network: str = config.NETWORK):
        self.reader = AlchemyBlockReader(api_key, network)
        self.api_key = api_key
        self.network = network
        self.pool_cache = {}
        self.last_cache_update_block = 0
        self.cache_file = Path(config.POOL_CACHE_FILE)

        # Timestamp cache optimization - avoid duplicate block timestamp fetching
        self.timestamp_cache = {}  # {block_number: timestamp}

        # Ensure cache directory exists
        self.cache_file.parent.mkdir(exist_ok=True)

        # Load existing cache
        self._load_pool_cache()

        # Initialize vault analyzers
        self._init_vault_analyzers()

    def _get_cached_timestamp(self, block_number: int) -> int:
        """Get block timestamp with caching to avoid duplicate RPC calls."""
        if block_number not in self.timestamp_cache:
            try:
                timestamp = self.reader.web3.eth.get_block(block_number).timestamp
                self.timestamp_cache[block_number] = timestamp
            except Exception as e:
                logging.warning(
                    f"Failed to get timestamp for block {block_number}: {e}"
                )
                return 0
        return self.timestamp_cache[block_number]

    def _vectorized_price_calculation(
        self, sqrt_prices_x96: List[int], token_symbols: List[str]
    ) -> List[float]:
        """Vectorized price calculation using numpy for better performance."""
        if not sqrt_prices_x96 or not token_symbols:
            return []

        # Convert to numpy arrays for vectorized operations
        sqrt_prices = np.array(sqrt_prices_x96, dtype=np.float64)

        # Vectorized calculation: price = (sqrtPriceX96 / 2^96)^2
        # Using numpy for optimized mathematical operations
        two_pow_96 = np.float64(2**96)
        price_ratios = np.power(sqrt_prices / two_pow_96, 2)

        # Determine which prices need inversion based on token addresses
        bera_address_lower = config.BERA_ADDRESS.lower()
        prices_in_bera = []

        for i, symbol in enumerate(token_symbols):
            if i >= len(price_ratios):
                prices_in_bera.append(0.0)
                continue

            cache_info = self.pool_cache.get(symbol, {})
            if "token_address" not in cache_info:
                prices_in_bera.append(0.0)
                continue

            token_address = cache_info["token_address"].lower()
            price_ratio = float(price_ratios[i])

            if token_address < bera_address_lower:
                # Token is token0, price is token1/token0 (BERA/token)
                price_in_bera = price_ratio
            else:
                # Token is token1, price is token0/token1 (BERA/token)
                price_in_bera = 1.0 / price_ratio if price_ratio > 0 else 0.0

            prices_in_bera.append(price_in_bera)

        return prices_in_bera

    def _load_pool_cache(self) -> None:
        """Load pool cache from file if it exists."""
        if not self.cache_file.exists():
            self.pool_cache = {}
            self.last_cache_update_block = 0
            return

        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self.pool_cache = data.get("pools", {})
                self.last_cache_update_block = data.get("last_update_block", 0)
            logging.info(f"Loaded pool cache from {self.cache_file}")
        except Exception as e:
            logging.warning(f"Failed to load pool cache: {e}")
            self.pool_cache = {}
            self.last_cache_update_block = 0

    def _save_pool_cache(self) -> None:
        """Save pool cache to file."""
        try:
            cache_data = {
                "pools": self.pool_cache,
                "last_update_block": self.last_cache_update_block,
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logging.debug(f"Saved pool cache to {self.cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save pool cache: {e}")

    def _init_vault_analyzers(self) -> None:
        """Initialize vault analyzer instances for APR calculations."""
        vault_info_list = get_all_vault_stake_tokens(self.api_key, self.network)
        self.reward_analyzers = []

        for vault_info in vault_info_list:
            analyzer = RewardVaultAnalyzer(
                api_key=self.api_key,
                network=self.network,
                vault_address=vault_info["vault_address"],
                lp_token_address=vault_info["stake_token"]["address"],
            )
            self.reward_analyzers.append(
                {
                    "analyzer": analyzer,
                    "vault_address": vault_info["vault_address"],
                    "symbol": vault_info["stake_token"]["symbol"],
                }
            )

    def _should_refresh_cache(self, current_block: int) -> bool:
        """Determine if cache should be refreshed based on block interval."""
        if not self.pool_cache:
            return True

        blocks_since_update = current_block - self.last_cache_update_block
        return blocks_since_update >= config.POOL_CACHE_REFRESH_BLOCKS

    def _update_pool_cache_from_subgraph(self, block_number: int) -> None:
        """Update pool cache by fetching latest pool addresses from subgraph."""
        logging.info(f"🔄 Updating pool cache from subgraph for block {block_number}")

        # Build GraphQL query for all LST tokens
        query_parts = []
        token_aliases = []

        for idx, lst in enumerate(config.LST_TOKENS):
            token_address = lst["address"].lower()
            bera_address = config.BERA_ADDRESS.lower()

            token0 = token_address
            token1 = bera_address
            if token0 > token1:
                token0, token1 = token1, token0

            alias = f"t{idx}"
            token_aliases.append((alias, lst))

            part = f"""
      {alias}: pools(where: {{token0: \"{token0}\", token1: \"{token1}\", liquidity_gt: \"0\"}}, orderBy: liquidity, orderDirection: desc, first: 1, block: {{number: $block}}) {{
        id
        token0 {{ id }}
        token1 {{ id }}
      }}"""
            query_parts.append(part)

        query_body = "query AllPools($block: Int!) {\n" + "\n".join(query_parts) + "\n}"
        variables = {"block": block_number}

        try:
            start_time = time.time()
            response = requests.post(
                config.GOLDSKY_ENDPOINT,
                json={"query": query_body, "variables": variables},
                timeout=config.SUBGRAPH_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            end_time = time.time()

            response_time_ms = (end_time - start_time) * 1000
            print(
                f"🔗 Pool cache update response time: {response_time_ms:.1f}ms (block {block_number})"
            )

            if "data" not in data:
                logging.warning(f"Subgraph query returned error: {data.get('errors')}")
                return

            # Update cache with pool addresses
            pool_data_map = data["data"]
            for alias, token_info in token_aliases:
                pools = pool_data_map.get(alias)
                if pools and len(pools) > 0:
                    pool_id = pools[0]["id"]
                    self.pool_cache[token_info["symbol"]] = {
                        "pool_address": pool_id,
                        "token_address": token_info["address"],
                        "updated_block": block_number,
                    }
                    logging.debug(f"Cached pool for {token_info['symbol']}: {pool_id}")

            self.last_cache_update_block = block_number
            self._save_pool_cache()

        except Exception as e:
            logging.error(f"Failed to update pool cache: {e}")

    def _prepare_batch_calls(self, block_number: int) -> List[Dict[str, Any]]:
        """Prepare batch contract calls for APR and price data."""
        calls = []
        call_id = 0

        # Prepare vault APR calls (reward rate + total supply)
        for analyzer_info in self.reward_analyzers:
            analyzer = analyzer_info["analyzer"]
            vault_address = analyzer_info["vault_address"]

            # Reward rate
            calls.append(
                {
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [
                        {
                            "to": vault_address,
                            "data": "0x7b0a47ee",  # rewardRate() function selector
                        },
                        hex(block_number),
                    ],
                    "id": call_id,
                }
            )
            call_id += 1

            # Total supply
            calls.append(
                {
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [
                        {
                            "to": analyzer.lp_token_address,
                            "data": "0x18160ddd",  # totalSupply() function selector
                        },
                        hex(block_number),
                    ],
                    "id": call_id,
                }
            )
            call_id += 1

        # Prepare LST token price calls (Uniswap V3 slot0)
        for cache_info in self.pool_cache.values():
            if "pool_address" in cache_info:
                pool_address = cache_info["pool_address"]
                calls.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "eth_call",
                        "params": [
                            {
                                "to": pool_address,
                                "data": "0x3850c7bd",  # slot0() function selector
                            },
                            hex(block_number),
                        ],
                        "id": call_id,
                    }
                )
                call_id += 1

        return calls

    def _process_batch_results(
        self, results: List[Dict], block_number: int, timestamp: int
    ) -> Dict[str, Any]:
        """Process batch call results into structured data."""
        processed_data = {
            "block": block_number,
            "timestamp": timestamp,
            "apr_data": {},
            "price_data": {},
        }

        result_index = 0

        # Process APR calculation results
        for analyzer_info in self.reward_analyzers:
            symbol = analyzer_info["symbol"]

            try:
                # Reward rate result with error checking
                if (
                    result_index < len(results)
                    and "result" in results[result_index]
                    and "error" not in results[result_index]
                ):
                    reward_rate_hex = results[result_index]["result"]
                    reward_rate = (
                        int(reward_rate_hex, 16) if reward_rate_hex != "0x" else 0
                    )
                else:
                    if result_index < len(results) and "error" in results[result_index]:
                        logging.debug(
                            f"Reward rate call failed for {symbol}: {results[result_index]['error']}"
                        )
                    reward_rate = 0
                result_index += 1

                # Total supply result with error checking
                if (
                    result_index < len(results)
                    and "result" in results[result_index]
                    and "error" not in results[result_index]
                ):
                    total_supply_hex = results[result_index]["result"]
                    total_supply = (
                        int(total_supply_hex, 16) if total_supply_hex != "0x" else 1
                    )
                else:
                    if result_index < len(results) and "error" in results[result_index]:
                        logging.debug(
                            f"Total supply call failed for {symbol}: {results[result_index]['error']}"
                        )
                    total_supply = 1
                result_index += 1

                # Calculate APR using same logic as RewardVaultAnalyzer
                if total_supply > 0:
                    reward_rate_per_second = reward_rate / config.PRECISION_E36
                    reward_rate_per_year = (
                        reward_rate_per_second
                        * config.SECONDS_PER_DAY
                        * config.DAYS_PER_YEAR
                    )
                    total_supply_normalized = total_supply / config.PRECISION
                    apr = (reward_rate_per_year / total_supply_normalized) * 100
                else:
                    apr = 0.0

                processed_data["apr_data"][symbol] = apr
                logging.debug(f"Block {block_number} {symbol} APR: {apr}%")

            except Exception as e:
                logging.warning(f"Failed to process APR for {symbol}: {e}")
                processed_data["apr_data"][symbol] = 0.0
                result_index += 2  # Skip both calls for this analyzer

        # Process price calculation results
        for symbol, cache_info in self.pool_cache.items():
            if "pool_address" not in cache_info:
                continue

            try:
                if (
                    result_index < len(results)
                    and "result" in results[result_index]
                    and "error" not in results[result_index]
                ):
                    slot0_hex = results[result_index]["result"]
                    if slot0_hex != "0x" and len(slot0_hex) >= 66:
                        # Extract sqrtPriceX96 (first 32 bytes)
                        sqrt_price_x96 = int(slot0_hex[2:66], 16)

                        if sqrt_price_x96 > 0:
                            # Calculate price from sqrtPriceX96
                            # price = (sqrtPriceX96 / 2^96)^2
                            price_ratio = (sqrt_price_x96 / (2**96)) ** 2

                            # Determine if we need to invert based on token order
                            token_address = cache_info["token_address"].lower()
                            bera_address = config.BERA_ADDRESS.lower()

                            if token_address < bera_address:
                                # Token is token0, price is token1/token0 (BERA/token)
                                price_in_bera = price_ratio
                            else:
                                # Token is token1, price is token0/token1 (BERA/token)
                                price_in_bera = (
                                    1.0 / price_ratio if price_ratio > 0 else 0.0
                                )
                        else:
                            price_in_bera = 0.0

                        processed_data["price_data"][symbol] = {
                            "price_in_bera": price_in_bera,
                            "pool_id": cache_info["pool_address"],
                        }
                        logging.debug(
                            f"Block {block_number} {symbol} price: {price_in_bera} BERA (sqrtPrice: {sqrt_price_x96})"
                        )
                    else:
                        processed_data["price_data"][symbol] = {
                            "price_in_bera": 0.0,
                            "pool_id": cache_info["pool_address"],
                        }
                        logging.debug(
                            f"Block {block_number} {symbol}: invalid slot0 response"
                        )
                else:
                    # Check if this was an RPC error
                    if result_index < len(results) and "error" in results[result_index]:
                        error_msg = (
                            results[result_index]
                            .get("error", {})
                            .get("message", "Unknown error")
                        )
                        logging.debug(
                            f"Block {block_number} {symbol} slot0 call failed: {error_msg}"
                        )
                    else:
                        logging.debug(f"Block {block_number} {symbol}: no slot0 result")

                    processed_data["price_data"][symbol] = {
                        "price_in_bera": 0.0,
                        "pool_id": cache_info["pool_address"],
                    }

                result_index += 1

            except Exception as e:
                logging.warning(f"Failed to process price for {symbol}: {e}")
                processed_data["price_data"][symbol] = {
                    "price_in_bera": 0.0,
                    "pool_id": cache_info.get("pool_address", ""),
                }
                result_index += 1

        return processed_data

    def collect_batch_data_at_block(
        self, block_number: int
    ) -> Optional[Dict[str, Any]]:
        """Collect APR and price data for specified block using batch calls."""
        try:
            # Get timestamp
            timestamp = self.reader.web3.eth.get_block(block_number).timestamp

            # Update pool cache if needed
            if self._should_refresh_cache(block_number):
                self._update_pool_cache_from_subgraph(block_number)

            # Prepare batch calls
            calls = self._prepare_batch_calls(block_number)

            if not calls:
                logging.warning(f"No batch calls prepared for block {block_number}")
                return None

            # Execute batch call
            start_time = time.time()
            rpc_url = config.ALCHEMY_RPC_URL_TEMPLATE.format(
                network=self.network, api_key=self.api_key
            )
            response = requests.post(
                rpc_url,
                json=calls,
                headers={"Content-Type": "application/json"},
                timeout=config.RPC_TIMEOUT,
            )
            response.raise_for_status()
            results = response.json()
            end_time = time.time()

            batch_time_ms = (end_time - start_time) * 1000
            print(
                f"⚡ Batch contract call time: {batch_time_ms:.1f}ms ({len(calls)} calls, block {block_number})"
            )

            # Process results with error checking
            if isinstance(results, list):
                # Check for individual RPC errors in batch response
                error_count = 0
                for i, result in enumerate(results):
                    if "error" in result:
                        error_count += 1
                        logging.warning(
                            f"Batch call {i} failed: {result.get('error', {}).get('message', 'Unknown error')}"
                        )

                if error_count > 0:
                    success_rate = ((len(results) - error_count) / len(results)) * 100
                    logging.warning(
                        f"Block {block_number}: {error_count}/{len(results)} batch calls failed "
                        f"(success rate: {success_rate:.1f}%)"
                    )

                # Process results even if some calls failed (with fallback values)
                return self._process_batch_results(results, block_number, timestamp)
            else:
                logging.warning(f"Unexpected batch response format: {results}")
                return None

        except Exception as e:
            logging.error(f"Failed to collect batch data for block {block_number}: {e}")
            return None

    def get_csv_row_data(self, block_number: int) -> Optional[List[Any]]:
        """Get CSV row data for specified block."""
        batch_data = self.collect_batch_data_at_block(block_number)
        if not batch_data:
            return None

        # Start with block and timestamp
        row_data = [batch_data["block"], batch_data["timestamp"]]

        # Add APR data for each vault
        for analyzer_info in self.reward_analyzers:
            symbol = analyzer_info["symbol"]
            apr = batch_data["apr_data"].get(symbol, 0.0)
            row_data.append(apr)

        # Add price data for each LST token
        for token in config.LST_TOKENS:
            symbol = token["symbol"]
            price_info = batch_data["price_data"].get(
                symbol, {"price_in_bera": 0.0, "pool_id": ""}
            )
            row_data.append(price_info["price_in_bera"])
            row_data.append(price_info["pool_id"])

        return row_data

    def get_multiple_csv_rows_data(
        self, block_numbers: List[int]
    ) -> List[Optional[List[Any]]]:
        """Get CSV row data for multiple blocks in a single mega-batch request."""
        if not block_numbers:
            return []

        # Prepare mega-batch calls for all blocks
        mega_batch_calls = []
        call_id = 0
        block_call_mapping = (
            {}
        )  # Maps call_id to (block_number, call_type, vault_index)

        for block_idx, block_number in enumerate(block_numbers):
            try:
                # Get timestamp for this block (cached to avoid duplicate RPC calls)
                timestamp = self._get_cached_timestamp(block_number)

                # Update pool cache if needed (check only for first block)
                if block_idx == 0 and self._should_refresh_cache(block_number):
                    self._update_pool_cache_from_subgraph(block_number)

                # Prepare vault APR calls for this block
                for vault_idx, analyzer_info in enumerate(self.reward_analyzers):
                    vault_address = analyzer_info["vault_address"]
                    analyzer = analyzer_info["analyzer"]

                    # Reward rate call
                    mega_batch_calls.append(
                        {
                            "jsonrpc": "2.0",
                            "method": "eth_call",
                            "params": [
                                {
                                    "to": vault_address,
                                    "data": "0x7b0a47ee",  # rewardRate()
                                },
                                hex(block_number),
                            ],
                            "id": call_id,
                        }
                    )
                    block_call_mapping[call_id] = (
                        block_number,
                        "reward_rate",
                        vault_idx,
                    )
                    call_id += 1

                    # Total supply call
                    mega_batch_calls.append(
                        {
                            "jsonrpc": "2.0",
                            "method": "eth_call",
                            "params": [
                                {
                                    "to": analyzer.lp_token_address,
                                    "data": "0x18160ddd",  # totalSupply()
                                },
                                hex(block_number),
                            ],
                            "id": call_id,
                        }
                    )
                    block_call_mapping[call_id] = (
                        block_number,
                        "total_supply",
                        vault_idx,
                    )
                    call_id += 1

                # Prepare LST token price calls for this block
                for token_idx, (symbol, cache_info) in enumerate(
                    self.pool_cache.items()
                ):
                    if "pool_address" in cache_info:
                        pool_address = cache_info["pool_address"]
                        mega_batch_calls.append(
                            {
                                "jsonrpc": "2.0",
                                "method": "eth_call",
                                "params": [
                                    {
                                        "to": pool_address,
                                        "data": "0x3850c7bd",  # slot0()
                                    },
                                    hex(block_number),
                                ],
                                "id": call_id,
                            }
                        )
                        block_call_mapping[call_id] = (
                            block_number,
                            "price",
                            token_idx,
                            symbol,
                        )
                        call_id += 1

            except Exception as e:
                logging.warning(
                    f"Failed to prepare calls for block {block_number}: {e}"
                )
                continue

        if not mega_batch_calls:
            return [None] * len(block_numbers)

        # Execute mega-batch call
        try:
            start_time = time.time()
            rpc_url = config.ALCHEMY_RPC_URL_TEMPLATE.format(
                network=self.network, api_key=self.api_key
            )
            response = requests.post(
                rpc_url,
                json=mega_batch_calls,
                headers={"Content-Type": "application/json"},
                timeout=config.MEGA_BATCH_TIMEOUT,
            )
            response.raise_for_status()
            results = response.json()
            end_time = time.time()

            batch_time_ms = (end_time - start_time) * 1000
            logging.info(
                f"⚡ Mega-batch call time: {batch_time_ms:.1f}ms "
                f"({len(mega_batch_calls)} calls, {len(block_numbers)} blocks)"
            )

        except Exception as e:
            logging.error(f"Mega-batch call failed: {e}")
            return [None] * len(block_numbers)

        # Process results and organize by block
        block_data = {}
        for block_number in block_numbers:
            timestamp = self._get_cached_timestamp(block_number)
            if timestamp > 0:
                block_data[block_number] = {
                    "block": block_number,
                    "timestamp": timestamp,
                    "apr_data": {},
                    "price_data": {},
                }
            else:
                block_data[block_number] = None

        # Process mega-batch results - separate price processing for vectorization
        error_count = 0
        price_results = {}  # {(block_number, symbol): sqrt_price_x96}

        for result in results:
            call_id = result.get("id")
            if call_id not in block_call_mapping:
                continue

            mapping_info = block_call_mapping[call_id]
            block_number = mapping_info[0]
            call_type = mapping_info[1]

            if block_data[block_number] is None:
                continue

            if "error" in result:
                error_count += 1
                continue

            try:
                if call_type == "reward_rate":
                    vault_idx = mapping_info[2]
                    symbol = self.reward_analyzers[vault_idx]["symbol"]
                    reward_rate_hex = result["result"]
                    reward_rate = (
                        int(reward_rate_hex, 16) if reward_rate_hex != "0x" else 0
                    )

                    if symbol not in block_data[block_number]["apr_data"]:
                        block_data[block_number]["apr_data"][symbol] = {
                            "reward_rate": 0,
                            "total_supply": 1,
                        }
                    block_data[block_number]["apr_data"][symbol][
                        "reward_rate"
                    ] = reward_rate

                elif call_type == "total_supply":
                    vault_idx = mapping_info[2]
                    symbol = self.reward_analyzers[vault_idx]["symbol"]
                    total_supply_hex = result["result"]
                    total_supply = (
                        int(total_supply_hex, 16) if total_supply_hex != "0x" else 1
                    )

                    if symbol not in block_data[block_number]["apr_data"]:
                        block_data[block_number]["apr_data"][symbol] = {
                            "reward_rate": 0,
                            "total_supply": 1,
                        }
                    block_data[block_number]["apr_data"][symbol][
                        "total_supply"
                    ] = total_supply

                elif call_type == "price":
                    symbol = mapping_info[3]
                    slot0_hex = result["result"]

                    if slot0_hex != "0x" and len(slot0_hex) >= 66:
                        sqrt_price_x96 = int(slot0_hex[2:66], 16)
                        if sqrt_price_x96 > 0:
                            # Store for vectorized processing
                            price_results[(block_number, symbol)] = sqrt_price_x96
                        else:
                            price_results[(block_number, symbol)] = 0

            except Exception as e:
                logging.warning(f"Failed to process result for call_id {call_id}: {e}")
                continue

        # Vectorized price processing optimization
        if price_results:
            # Group by unique tokens for batch processing
            unique_symbols = list(set(symbol for _, symbol in price_results.keys()))

            # Process prices in batches per symbol
            for symbol in unique_symbols:
                # Get all sqrt prices for this symbol
                symbol_prices = []
                symbol_block_mapping = []

                for (block_number, sym), sqrt_price in price_results.items():
                    if sym == symbol:
                        symbol_prices.append(sqrt_price)
                        symbol_block_mapping.append(block_number)

                if symbol_prices:
                    # Vectorized calculation for this symbol
                    calculated_prices = self._vectorized_price_calculation(
                        symbol_prices, [symbol] * len(symbol_prices)
                    )

                    # Apply results back to block data
                    for i, block_number in enumerate(symbol_block_mapping):
                        if i < len(calculated_prices):
                            price_in_bera = calculated_prices[i]
                            block_data[block_number]["price_data"][symbol] = {
                                "price_in_bera": price_in_bera,
                                "pool_id": self.pool_cache[symbol]["pool_address"],
                            }

        # Calculate APRs and build CSV rows
        csv_rows = []
        for block_number in block_numbers:
            if block_data[block_number] is None:
                csv_rows.append(None)
                continue

            try:
                # Calculate APRs
                for symbol, apr_data in block_data[block_number]["apr_data"].items():
                    reward_rate = apr_data.get("reward_rate", 0)
                    total_supply = apr_data.get("total_supply", 1)

                    if total_supply > 0:
                        reward_rate_per_second = reward_rate / config.PRECISION_E36
                        reward_rate_per_year = (
                            reward_rate_per_second
                            * config.SECONDS_PER_DAY
                            * config.DAYS_PER_YEAR
                        )
                        total_supply_normalized = total_supply / config.PRECISION
                        apr = (reward_rate_per_year / total_supply_normalized) * 100
                    else:
                        apr = 0.0

                    block_data[block_number]["apr_data"][symbol] = apr

                # Build CSV row
                row_data = [
                    block_data[block_number]["block"],
                    block_data[block_number]["timestamp"],
                ]

                # Add APR data
                for analyzer_info in self.reward_analyzers:
                    symbol = analyzer_info["symbol"]
                    apr = block_data[block_number]["apr_data"].get(symbol, 0.0)
                    row_data.append(apr)

                # Add price data
                for token in config.LST_TOKENS:
                    symbol = token["symbol"]
                    price_info = block_data[block_number]["price_data"].get(
                        symbol, {"price_in_bera": 0.0, "pool_id": ""}
                    )
                    row_data.append(price_info["price_in_bera"])
                    row_data.append(price_info["pool_id"])

                csv_rows.append(row_data)

            except Exception as e:
                logging.warning(
                    f"Failed to build CSV row for block {block_number}: {e}"
                )
                csv_rows.append(None)

        if error_count > 0:
            success_rate = ((len(results) - error_count) / len(results)) * 100
            logging.warning(
                f"Mega-batch: {error_count}/{len(results)} calls failed (success rate: {success_rate:.1f}%)"
            )

        return csv_rows
