from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict

import requests
from web3 import Web3

from src import config
from src.abi import ERC20 as erc20
from src.abi.RewardVault import abi as reward_vault_abi
from src.analysis.calculate_lp_value import LPValueCalculator
from src.blockchain.blockchain_reader import AlchemyBlockReader


class RewardVaultAnalyzer:
    """
    Analyzes RewardVault APR and related LP token values for a specific vault.
    """

    def __init__(
        self,
        api_key: str,
        network: str,
        vault_address: str,
        lp_token_address: str,  # LP token associated with this vault
    ):
        """
        Initializes the analyzer for a specific reward vault.

        Parameters
        ----------
        api_key : str
            Alchemy API key.
        network : str
            Blockchain network (e.g., 'berachain-mainnet').
        vault_address : str
            The address of the RewardVault contract.
        lp_token_address : str
            The address of the LP token staked in the vault.
        """
        self.reader = AlchemyBlockReader(api_key, network=network)
        self.lp_calculator = LPValueCalculator(api_key, network=network)
        self.vault_address = vault_address
        self.lp_token_address = (
            lp_token_address  # Store the specific LP token for this vault
        )

    def _get_block_timestamp(self, block_number: int) -> datetime:
        """Gets timestamp for a block number using the internal reader."""
        block = self.reader.web3.eth.get_block(block_number)
        return datetime.fromtimestamp(block.timestamp)

    def _get_pool_data_at_block(self, block_number: int) -> Dict[str, Any] | None:
        """
        Fetches pool data for BERA/HONEY from subgraph at a specific block.
        Note: This is specific to LP_ADDRESS for now.
        If the vault is for a different LP, this needs generalization or removal if LP value is fetched differently.
        """
        token0 = config.BERA_ADDRESS.lower()
        token1 = (
            config.HONEY_ADDRESS.lower()
        )  # Assuming BERA/HONEY for LP value context

        if token0 > token1:
            token0, token1 = token1, token0

        logging.debug(
            f"Subgraph: Querying BERA/HONEY pool at block {block_number} for {self.lp_token_address}"
        )

        query = """
        query GetPoolDataAtBlock($blockNumber: Int!, $token0: String!, $token1: String!) {
          pools(
            where: {token0: $token0, token1: $token1, liquidity_gt: "0"},
            orderBy: liquidity, orderDirection: desc, first: 1,
            block: {number: $blockNumber}
          ) {
            id token0 { id symbol decimals } token1 { id symbol decimals }
            feeTier liquidity sqrtPrice tick token0Price token1Price
            totalValueLockedToken0 totalValueLockedToken1
          }
        }
        """
        variables = {"blockNumber": block_number, "token0": token0, "token1": token1}
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    config.GOLDSKY_ENDPOINT,
                    json={"query": query, "variables": variables},
                )
                response.raise_for_status()
                data = response.json()
                if (
                    data.get("data")
                    and data["data"].get("pools")
                    and len(data["data"]["pools"]) > 0
                ):
                    logging.debug(
                        f"Subgraph: Pool data success {data['data']['pools'][0]['id']}"
                    )
                    return data["data"]["pools"][0]
                logging.warning(
                    f"Subgraph query for {self.lp_token_address}@{block_number} returned no pool data or error: {data.get('errors')}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logging.error(
                    f"Subgraph connection error for {self.lp_token_address}@{block_number}: {e}",
                    exc_info=True,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logging.error(
                    f"Error processing subgraph response for {self.lp_token_address}@{block_number}: {e}",
                    exc_info=True,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        return None

    def _calculate_lp_token_value_bera_at_block(self, block_number: int) -> float:
        """
        Calculates the BERA value of 1 LP token (self.lp_token_address)
        staked in THIS vault (self.vault_address) at a specific block.
        """
        try:
            # LPValueCalculator is now block-aware.
            lp_value_bera = self.lp_calculator.calculate_lp_value_in_bera(
                self.lp_token_address, block_number=block_number
            )
            if lp_value_bera is not None:
                return lp_value_bera
            logging.warning(
                f"Could not determine LP value for {self.lp_token_address} at block {block_number}"
            )
            return 0.0  # Default if LP value cannot be determined
        except Exception as e:
            logging.error(
                f"Error calculating LP value for {self.lp_token_address} at block {block_number}: {e}",
                exc_info=True,
            )
            return 0.0

    def get_apr_details_at_block(
        self, block_number: int | None = None
    ) -> Dict[str, Any]:
        """
        Calculates APR details for the vault at a specific block.
        Returns a dictionary containing APR and related metrics.
        """
        if block_number is None:
            block_number = self.reader.get_latest_block_number()

        block_ts = self._get_block_timestamp(block_number)

        logging.debug(
            f"APR Calc: Block {block_number} ({block_ts.strftime('%Y-%m-%d %H:%M:%S')}) for vault {self.vault_address}"
        )

        # Batch fetch rewardRate and totalSupply in a single RPC call
        reward_rate_raw, total_supply_staked_raw = self.reader.batch_call_read_function(
            [
                {
                    "contract_address": self.vault_address,
                    "abi": reward_vault_abi,
                    "function_name": "rewardRate",
                    "function_params": [],
                },
                {
                    "contract_address": self.vault_address,
                    "abi": reward_vault_abi,
                    "function_name": "totalSupply",
                    "function_params": [],
                },
            ],
            block_number=block_number,
        )

        reward_per_second = reward_rate_raw / config.PRECISION_E36

        total_supply_staked_adjusted = total_supply_staked_raw / config.PRECISION

        logging.debug(
            f"Vault Info: Address={self.vault_address}, RewardRate={reward_per_second:.8f} BGT/sec, StakedLP={total_supply_staked_adjusted:.4f}"
        )

        # Use the new method for LP token value
        lp_token_value_bera = self._calculate_lp_token_value_bera_at_block(block_number)

        logging.debug(
            f"LP Token Info: Address={self.lp_token_address}, ValueInBERA={lp_token_value_bera:.8f} at block {block_number}"
        )

        # APR Calculation (moved from standalone calculate_apr)
        annual_reward_bgt = (
            reward_per_second * config.SECONDS_PER_DAY * config.DAYS_PER_YEAR
        )
        total_staked_value_bera = total_supply_staked_adjusted * lp_token_value_bera

        apr_percentage = 0.0
        if total_staked_value_bera > 0:
            apr_percentage = (annual_reward_bgt / total_staked_value_bera) * 100

        reward_per_day = reward_per_second * config.SECONDS_PER_DAY
        logging.debug(
            f"Reward Info: Daily={reward_per_day:.2f} BGT, Annual={annual_reward_bgt:.2f} BGT"
        )
        logging.debug(f"Stake Value: Total={total_staked_value_bera:.2f} BERA")
        logging.debug(
            f"Calculated APR for vault {self.vault_address}: {apr_percentage:.2f}%"
        )

        return {
            "block_number": block_number,
            "timestamp": block_ts,
            "reward_rate_raw": reward_rate_raw,
            "total_supply_staked_raw": total_supply_staked_raw,
            "lp_token_value_bera": lp_token_value_bera,
            "apr_percentage": apr_percentage,  # APR as percentage, e.g., 123.45
        }


# This module is now intended to be imported.
