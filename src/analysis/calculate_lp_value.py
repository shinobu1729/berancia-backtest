import logging

import requests
from web3 import Web3

from src import config
from src.abi import ERC20 as erc20
from src.abi import KodiakIslandWithRouter as kodiak_island
from src.blockchain.blockchain_reader import AlchemyBlockReader


class LPValueCalculator:
    """
    LPトークンのBera建て価格を計算するクラス
    """

    def __init__(self, api_key, network=config.NETWORK):
        """
        初期化
        api_key: AlchemyのAPIキー
        network: 対象ネットワーク
        """
        self.reader = AlchemyBlockReader(api_key, network=network)
        self.web3 = self.reader.web3
        logging.debug(
            f"LPValueCalculator initialized for network: {network}, Web3 connected: {self.web3.is_connected()}"
        )

    def get_pools_from_subgraph(
        self, token0_address, token1_address, block_number: int | None = None
    ):
        """
        サブグラフから2つのトークン間のプール情報を取得
        """
        # アドレスを小文字に変換
        token0_address = token0_address.lower()
        token1_address = token1_address.lower()

        # 辞書順でtoken0とtoken1をソート
        if token0_address > token1_address:
            token0_address, token1_address = token1_address, token0_address

        logging.debug(
            f"Subgraph: Searching pool for token0={token0_address}, token1={token1_address}, block: {block_number}"
        )

        query = """
        query GetPoolsByTokens($token0: String!, $token1: String!, $block: Block_height) {
          pools(where: {
            token0: $token0,
            token1: $token1,
            liquidity_gt: "0"
          }, orderBy: liquidity, orderDirection: desc, first: 1, block: $block) {
            id
            token0 {
              id
              symbol
              decimals
            }
            token1 {
              id
              symbol
              decimals
            }
            feeTier
            liquidity
            token0Price
            token1Price
            volumeToken0
            volumeToken1
            volumeUSD
          }
        }
        """

        variables = {"token0": token0_address, "token1": token1_address}

        try:
            response = requests.post(
                config.GOLDSKY_ENDPOINT, json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data and "pools" in data["data"]:
                return data["data"]["pools"]
            else:
                logging.warning(
                    f"Subgraph query for {token0_address}/{token1_address}@{block_number} returned no pool data or error: {data.get('errors')}"
                )
                return []
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Subgraph connection error for {token0_address}/{token1_address}@{block_number}: {e}",
                exc_info=True,
            )
            return []
        except Exception as e:  # Catch other potential errors like JSONDecodeError
            logging.error(
                f"Error processing subgraph response for {token0_address}/{token1_address}@{block_number}: {e}",
                exc_info=True,
            )
            return []

    def get_token_price_in_bera(
        self, token_address: str, block_number: int | None = None
    ):
        """
        トークンのBera建て価格を取得
        """
        # アドレスを小文字に変換
        token_address = token_address.lower()

        # BERAの場合は価格 = 1
        if token_address == config.BERA_ADDRESS:
            return 1.0

        # BERAとの直接プールから価格を取得
        pools = self.get_pools_from_subgraph(
            token_address, config.BERA_ADDRESS, block_number=block_number
        )

        if pools:
            pool = pools[0]  # 最も流動性の高いプール

            # トークンがtoken0かtoken1かを確認
            is_token_token0 = pool["token0"]["id"].lower() == token_address

            # 価格を取得
            if is_token_token0:
                price = float(pool["token1Price"])
            else:
                price = float(pool["token0Price"])

            logging.debug(
                f"Token {token_address} price (BERA) via direct pool: {price} at block {block_number}"
            )
            return price

        # 価格が取得できない場合
        logging.warning(
            f"Could not get price for token {token_address} at block {block_number} via direct BERA pool."
        )
        return None

    def get_token_pool_data(self, lp_address: str, block_number: int | None = None):
        """
        LPトークンからtoken0/token1アドレスと残高、総供給量を取得
        """
        try:
            # KodiakIslandWithRouter ABIを使用して直接データを取得
            lp_address = lp_address.lower()

            # --- Batch RPC: token0, token1, balances, totalSupply ---
            (
                token0_address,
                token1_address,
                underlying_balances,
                total_supply,
            ) = self.reader.batch_call_read_function(
                [
                    {
                        "contract_address": lp_address,
                        "abi": kodiak_island.abi,
                        "function_name": "token0",
                        "function_params": [],
                    },
                    {
                        "contract_address": lp_address,
                        "abi": kodiak_island.abi,
                        "function_name": "token1",
                        "function_params": [],
                    },
                    {
                        "contract_address": lp_address,
                        "abi": kodiak_island.abi,
                        "function_name": "getUnderlyingBalances",
                        "function_params": [],
                    },
                    {
                        "contract_address": lp_address,
                        "abi": kodiak_island.abi,
                        "function_name": "totalSupply",
                        "function_params": [],
                    },
                ],
                block_number=block_number,
            )

            token0_balance = underlying_balances[0]
            token1_balance = underlying_balances[1]

            # --- Batch decimals for token0 / token1 ---
            try:
                token0_decimals, token1_decimals = self.reader.batch_call_read_function(
                    [
                        {
                            "contract_address": token0_address,
                            "abi": erc20.abi,
                            "function_name": "decimals",
                            "function_params": [],
                        },
                        {
                            "contract_address": token1_address,
                            "abi": erc20.abi,
                            "function_name": "decimals",
                            "function_params": [],
                        },
                    ],
                    block_number=block_number,
                )
            except Exception as e:
                logging.warning(
                    f"Could not get decimals for {token0_address} or {token1_address}, defaulting to 18. Error: {e}"
                )
                token0_decimals = 18
                token1_decimals = 18

            logging.debug(
                f"LP Data ({lp_address}@{block_number}): T0={token0_address}, T1={token1_address}, Bal0={token0_balance}, Bal1={token1_balance}, Supply={total_supply}"
            )

            return {
                "token0_address": token0_address,
                "token1_address": token1_address,
                "token0_balance": token0_balance,
                "token1_balance": token1_balance,
                "token0_decimals": token0_decimals,
                "token1_decimals": token1_decimals,
                "total_supply": total_supply,
            }

        except Exception as e:
            logging.error(
                f"Error getting pool data for {lp_address}@{block_number}: {e}",
                exc_info=True,
            )
            import traceback

            traceback.print_exc()
            return None

    def calculate_lp_value(self, lp_address: str, block_number: int | None = None):
        """
        LPトークンの情報と価値を計算する共通関数
        """
        try:
            logging.debug(f"LP価値計算開始: {lp_address} at block {block_number}")
            lp_address = lp_address.lower()

            # トークンプールのデータを取得
            pool_data = self.get_token_pool_data(lp_address, block_number=block_number)
            if not pool_data:
                raise Exception("プールデータを取得できませんでした")

            token0_address = pool_data["token0_address"]
            token1_address = pool_data["token1_address"]
            token0_balance = pool_data["token0_balance"]
            token1_balance = pool_data["token1_balance"]
            token0_decimals = pool_data["token0_decimals"]
            token1_decimals = pool_data["token1_decimals"]
            total_supply = pool_data["total_supply"]

            # トークンのBera建て価格を取得
            token0_price = self.get_token_price_in_bera(
                token0_address, block_number=block_number
            )
            token1_price = self.get_token_price_in_bera(
                token1_address, block_number=block_number
            )

            # 小数点を考慮したトークン残高価値計算
            token0_value_in_bera = (token0_balance * token0_price) / (
                10**token0_decimals
            )
            token1_value_in_bera = (token1_balance * token1_price) / (
                10**token1_decimals
            )
            if config.DEBUG:
                print(f"DEBUG: token0_value_in_bera: {token0_value_in_bera}")
                print(f"DEBUG: token1_value_in_bera: {token1_value_in_bera}")

            # プール全体のBera建て価値
            total_value_in_bera = token0_value_in_bera + token1_value_in_bera

            # 1 LPトークンあたりの価値
            lp_token_value_in_bera = (
                total_value_in_bera * config.PRECISION / total_supply
                if total_supply > 0
                else 0
            )

            return {
                "lp_address": lp_address,
                "token0_address": token0_address,
                "token1_address": token1_address,
                "token0_balance": token0_balance,
                "token1_balance": token1_balance,
                "token0_decimals": token0_decimals,
                "token1_decimals": token1_decimals,
                "token0_price_in_bera": token0_price,
                "token1_price_in_bera": token1_price,
                "total_supply": total_supply,
                "total_value_in_bera": total_value_in_bera,
                "lp_token_value_in_bera": lp_token_value_in_bera,
            }

        except Exception as e:
            logging.error(
                f"LP価値計算エラー ({lp_address}@{block_number}): {e}", exc_info=True
            )
            import traceback

            traceback.print_exc()
            return None

    def calculate_lp_value_in_bera(
        self, lp_address: str, block_number: int | None = None
    ):
        """
        LPトークンの1トークンあたりのBera建て価格を計算

        Parameters:
        lp_address (str): LPトークンのアドレス
        block_number (int | None): Optional block number for historical calculation.

        Returns:
        float: 1 LPトークンあたりのBera建て価格
        None: 計算エラー時
        """
        result = self.calculate_lp_value(lp_address, block_number=block_number)
        if result:
            return result["lp_token_value_in_bera"]
        return None


def test():
    """
    サンプル実行コード
    """
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    calculator = LPValueCalculator(config.ALCHEMY_API_KEY)  # Uses config for network

    # 各LPトークンの価値を計算
    for lp_address in [config.LP_ADDRESS]:
        print(f"\n===== {lp_address} の分析 =====")

        # 詳細情報取得
        lp_value = calculator.calculate_lp_value(lp_address)
        if lp_value:
            print(f"LP トークン: {lp_value['lp_address']}")
            print(f"token0: {lp_value['token0_address']}")
            print(f"token0 残高: {lp_value['token0_balance']}")
            print(f"token0 価格 (BERA): {lp_value['token0_price_in_bera']}")
            print(f"token1: {lp_value['token1_address']}")
            print(f"token1 残高: {lp_value['token1_balance']}")
            print(f"token1 価格 (BERA): {lp_value['token1_price_in_bera']}")
            print(f"総供給量: {lp_value['total_supply']}")
            print(f"プール総価値 (BERA): {lp_value['total_value_in_bera']}")
            print(f"1 LPトークンあたりの価値 (BERA): {lp_value['lp_token_value_in_bera']}")
        else:
            print(f"LP価値の計算に失敗しました: {lp_address}")

        # 個別関数利用例
        print("\n簡易関数使用例:")
        bera_value = calculator.calculate_lp_value_in_bera(lp_address)
        print(f"1 LPトークンあたりの価値 (BERA): {bera_value}")


if __name__ == "__main__":
    test()
