import json
import logging
from typing import Any, Dict, List, Optional, Union

import requests
from web3 import Web3
from web3.exceptions import ContractLogicError


class AlchemyBlockReader:
    """
    Alchemy APIを使用して過去のブロックの状態を読み取るクラス
    """

    def __init__(self, api_key: str, network: str = "eth-mainnet"):
        """
        初期化メソッド

        Args:
            api_key: Alchemy API Key
            network: ネットワーク名 (例: "eth-mainnet", "polygon-mainnet")
        """
        self.api_key = api_key
        self.network = network

        # Import config here to avoid circular imports
        from src import config

        self.base_url = config.ALCHEMY_RPC_URL_TEMPLATE.format(
            network=network, api_key=api_key
        )
        self.web3 = Web3(Web3.HTTPProvider(self.base_url))

    def get_contract(self, contract_address: str, abi: Union[str, List, Dict]) -> Any:
        """
        コントラクトインスタンスを取得

        Args:
            contract_address: コントラクトアドレス
            abi: コントラクトのABI (JSON文字列、リスト、または辞書)

        Returns:
            Contract: コントラクトインスタンス
        """
        if isinstance(abi, str):
            abi = json.loads(abi)

        return self.web3.eth.contract(
            address=self.web3.to_checksum_address(contract_address), abi=abi
        )

    def call_read_function(
        self,
        contract_address: str,
        abi: Union[str, List, Dict],
        function_name: str,
        function_params: Optional[List] = None,
        block_number: Optional[int] = None,
        block_hash: Optional[str] = None,
    ) -> Any:
        """
        読み取り関数の呼び出し

        Args:
            contract_address: コントラクトアドレス
            abi: コントラクトのABI
            function_name: 呼び出す関数名
            function_params: 関数パラメータ (デフォルト: None)
            block_number: 特定のブロック番号 (デフォルト: None)
            block_hash: 特定のブロックハッシュ (デフォルト: None)

        Returns:
            Any: 関数の戻り値
        """
        if function_params is None:
            function_params = []

        contract = self.get_contract(contract_address, abi)
        function = getattr(contract.functions, function_name)

        # ブロック識別子を準備
        block_identifier = "latest"
        if block_number is not None:
            block_identifier = block_number
        elif block_hash is not None:
            block_identifier = block_hash

        try:
            return function(*function_params).call(block_identifier=block_identifier)
        except ContractLogicError as e:
            print(f"コントラクト呼び出しエラー: {e}")
            raise
        except Exception as e:
            print(f"エラー: {e}")
            raise

    def get_block_timestamp(self, block_number: int) -> int:
        """
        指定したブロック番号のタイムスタンプを取得

        Args:
            block_number: ブロック番号

        Returns:
            int: ブロックのタイムスタンプ（Unix時間）
        """
        block = self.web3.eth.get_block(block_number)
        return block.timestamp

    def get_latest_block_number(self) -> int:
        """
        最新のブロック番号を取得

        Returns:
            int: 最新のブロック番号
        """
        return self.web3.eth.block_number

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------

    def batch_call_read_function(
        self, calls: List[Dict], block_number: Optional[int] = None
    ):
        """Perform multiple eth_call requests in a single JSON-RPC batch.

        Parameters
        ----------
        calls : List[Dict]
            Each dict must contain:
              - contract_address (str)
              - abi (list | dict | str)
              - function_name (str)
              - function_params (List)  optional
        block_number : Optional[int]
            Block number for all calls (defaults to latest if None)

        Returns
        -------
        List[Any]
            Decoded return values for each call in the same order as requests.
        """

        if block_number is None:
            block_tag = "latest"
        else:
            block_tag = hex(block_number)

        batch_payload = []
        decoders = []  # store (id, decode_fn)

        for idx, item in enumerate(calls):
            contract_address = item["contract_address"]
            abi = item["abi"]
            fname = item["function_name"]
            fparams = item.get("function_params", []) or []

            contract = self.get_contract(contract_address, abi)
            fn_obj = getattr(contract.functions, fname)(*fparams)

            # encode call data (use build_transaction for compatibility with Web3 v6+)
            try:
                data = fn_obj.build_transaction({"gas": 0})["data"]
            except Exception:
                # Fallback for older Web3 versions
                data = fn_obj._encode_transaction_data()

            batch_payload.append(
                {
                    "jsonrpc": "2.0",
                    "id": idx,
                    "method": "eth_call",
                    "params": [
                        {"to": contract_address, "data": data},
                        block_tag,
                    ],
                }
            )

            # Capture decode routine
            def _simplify(v):
                # Return single value if result is a 1-element tuple
                return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v

            def _decoder(hex_data, fn=fn_obj, codec=self.web3.codec):
                if hex_data is None:
                    return None

                # -- A. First try decode_output if available (Web3 v5) --
                if hasattr(fn, "decode_output"):
                    try:
                        decoded = fn.decode_output(hex_data)
                    except Exception:
                        pass
                    else:
                        return _simplify(decoded)

                # -- B. Fallback to manual decoding --
                bin_data = Web3.to_bytes(hexstr=hex_data)
                output_types = [o["type"] for o in fn.abi.get("outputs", [])]

                # Handle both Web3 v5 and v6
                if hasattr(codec, "decode_abi"):
                    decoded = codec.decode_abi(output_types, bin_data)  # v5
                else:
                    decoded = codec.decode(output_types, bin_data)  # v6

                return _simplify(decoded)

            decoders.append(_decoder)

        # Send batch request using requests to bypass Web3 missing batch helper
        try:
            response = requests.post(self.base_url, json=batch_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Batch RPC call failed: {e}")

        # Map id -> result
        id_map = {item["id"]: item for item in data}

        results = []
        for idx, decoder in enumerate(decoders):
            item = id_map.get(idx)

            # Handle missing or error cases
            if item is None:
                logging.error(f"Batch RPC: missing response for id {idx}")
                results.append(None)
                continue

            if "error" in item:
                logging.error(
                    f"Batch RPC error id {idx}: {item['error']}"  # type: ignore
                )
                raise RuntimeError(f"Batch RPC error: {item['error']}")

            if "result" not in item:
                logging.error(f"Batch RPC: no result field for id {idx}")
                results.append(None)
                continue

            results.append(decoder(item["result"]))

        return results


# === Utility Function =========================================================


def get_block_timestamp(block_number: int) -> int:
    """Return block timestamp (unix seconds) for the specified block on the configured network."""
    from src import config  # Adjusted import path

    reader = AlchemyBlockReader(config.ALCHEMY_API_KEY, network=config.NETWORK)
    return reader.get_block_timestamp(block_number)
