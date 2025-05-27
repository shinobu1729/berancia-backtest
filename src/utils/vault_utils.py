#!/usr/bin/env python3
"""
vault_utils.py
==============
Utility functions for RewardVault operations.
"""

import logging

from web3 import Web3

from src import config
from src.abi.ERC20 import abi as erc20_abi
from src.abi.RewardVault import abi as reward_vault_abi
from src.blockchain.blockchain_reader import AlchemyBlockReader


def get_stake_token_info(
    vault_address: str, api_key: str = None, network: str = None
) -> dict:
    """
    Get stake token information from a RewardVault.

    Args:
        vault_address: RewardVault contract address
        api_key: Alchemy API key (uses config if None)
        network: Network name (uses config if None)

    Returns:
        dict: {"address": stake_token_address, "symbol": stake_token_symbol}
    """
    if api_key is None:
        api_key = config.ALCHEMY_API_KEY
    if network is None:
        network = config.NETWORK

    reader = AlchemyBlockReader(api_key, network)
    web3 = reader.web3

    # Create RewardVault contract instance
    vault_contract = web3.eth.contract(
        address=Web3.to_checksum_address(vault_address), abi=reward_vault_abi
    )

    try:
        # Call stakeToken() to get LP token address
        stake_token_address = vault_contract.functions.stakeToken().call()
        logging.debug(f"Vault {vault_address} stake token: {stake_token_address}")

        # Get LP token symbol
        lp_contract = web3.eth.contract(
            address=Web3.to_checksum_address(stake_token_address), abi=erc20_abi
        )

        stake_token_symbol = lp_contract.functions.symbol().call()
        logging.debug(f"Stake token {stake_token_address} symbol: {stake_token_symbol}")

        return {"address": stake_token_address, "symbol": stake_token_symbol}

    except Exception as e:
        logging.error(f"Failed to get stake token info for vault {vault_address}: {e}")
        raise


def get_all_vault_stake_tokens(api_key: str = None, network: str = None) -> list:
    """
    Get stake token information for all configured RewardVaults.

    Args:
        api_key: Alchemy API key (uses config if None)
        network: Network name (uses config if None)

    Returns:
        list: [{"vault_address": vault_addr, "stake_token": {"address": addr, "symbol": symbol}}, ...]
    """
    vault_info = []

    for vault_address in config.REWARD_VAULT_ADDRESSES:
        try:
            stake_token_info = get_stake_token_info(vault_address, api_key, network)
            vault_info.append(
                {"vault_address": vault_address, "stake_token": stake_token_info}
            )
            logging.info(
                f"Vault {vault_address}: stake token {stake_token_info['symbol']} ({stake_token_info['address']})"
            )
        except Exception as e:
            logging.error(f"Failed to get info for vault {vault_address}: {e}")
            continue

    return vault_info
