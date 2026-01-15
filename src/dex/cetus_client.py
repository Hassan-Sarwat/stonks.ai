"""
Cetus DEX Client for interacting with the Cetus DEX on the SUI blockchain.
"""

import time
import requests
from typing import Dict, Any, List, Optional
from decimal import Decimal

from src.blockchain.sui_client import SuiBlockchainClient


class CetusDexClient:
    def __init__(self, sui_client: SuiBlockchainClient):
        """
        Initialize Cetus DEX client.

        Args:
            sui_client: Initialized SUI blockchain client
        """
        self.sui_client = sui_client

        # Cetus contract addresses (these would need to be updated based on
        # deployment)
        if sui_client.network == "mainnet":
            self.pool_ids = {
                "SUI_USDC": (
                    "0x5eb2dfcdd1b15c8d3a8d6c3b3d95065bddd283899fa17bfcf36c7"
                    "cbd6be0f454"
                ),
                # Add other pool IDs as needed
            }
            self.router_id = (
                "0x3a5143bb1196e3bcdfab6203d1683ae29edd26294fc8bfeafe4aaa9d"
                "2704df37"
            )
        else:  # testnet
            self.pool_ids = {
                "SUI_USDC": (
                    "0x7e9e0d2a84d87cd26196c2c205e334c4849cbe3dd9250d1113fb7"
                    "dd4e8e38d8f"
                ),
                # Add testnet pool IDs
            }
            self.router_id = (
                "0x2eeaab737b37137b94bfa8f841f92e36a153641119da3456dec19275"
                "14111bc4"
            )
        # Token addresses
        if sui_client.network == "mainnet":
            self.tokens = {
                "SUI": "0x2::sui::SUI",
                "USDC": (
                    "0x5d4b302506645c37ff133b98c4b50a5ae14841659738d6d733d59"
                    "d0d217a93bf::coin::COIN"
                ),
            }
        else:  # testnet
            self.tokens = {
                "SUI": "0x2::sui::SUI",
                "USDC": (
                    "0x7016aae72571c1b4e458be6a3b15d252c9b227343708ceb8d0f5"
                    "bfb3c1cd5a8d::coin::COIN"
                ),
            }

    async def get_pool_info(self, pool_pair: str) -> Dict[str, Any]:
        """Get information about a specific liquidity pool."""
        pool_id = self.pool_ids.get(pool_pair)
        if not pool_id:
            raise ValueError(f"Pool pair {pool_pair} not found")

        pool_data = await self.sui_client.get_object(pool_id)
        return pool_data

    async def get_price(self, token_pair: str) -> Decimal:
        """
        Get the current price for a token pair from CoinGecko.
        """
        print(f"Getting price for {token_pair}...")

        # Parse token pair
        tokens = token_pair.split("_")
        if len(tokens) != 2:
            raise ValueError(f"Invalid token pair format: {token_pair}")

        # Parse tokens but we don't need them right now
        # base_token, quote_token = tokens[0], tokens[1]

        try:
            coingecko_url = (
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=sui&vs_currencies=usd"
            )
            
            # Make API request to CoinGecko
            print(f"Fetching price from CoinGecko for {token_pair}...")
            cg_response = requests.get(coingecko_url)
            cg_response.raise_for_status()
            cg_data = cg_response.json()
            
            if "sui" in cg_data and "usd" in cg_data["sui"]:
                price = Decimal(str(cg_data["sui"]["usd"]))
                print(
                    f"Fetched price from CoinGecko for {token_pair}: "
                    f"${price}"
                )
                return price

            # If we couldn't get the price from the API, use a fallback
            print(
                f"Could not find price for {token_pair} in API response, "
                f"using fallback"
            )
            if token_pair == "SUI_USDC":
                return Decimal("0.85")  # $0.85 per SUI as fallback
            else:
                return Decimal("1.00")

        except Exception as e:
            print(f"Error fetching price data: {e}")
            # Fallback to default prices in case of API failure
            if token_pair == "SUI_USDC":
                return Decimal("0.85")  # $0.85 per SUI
            else:
                return Decimal("1.00")

    async def build_swap_transaction(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        deadline: Optional[int] = None,
    ) -> str:
        """
        Build a swap transaction.

        Args:
            token_in: Token to swap from (e.g., "SUI")
            token_out: Token to swap to (e.g., "USDC")
            amount_in: Amount of token_in to swap
            min_amount_out: Minimum amount of token_out to receive
            deadline: Transaction deadline in seconds from now

        Returns:
            Transaction bytes ready to be signed and executed
        """
        if not deadline:
            deadline = int(time.time()) + 300  # 5 minutes from now

        # Get token addresses
        token_in_address = self.tokens.get(token_in)
        token_out_address = self.tokens.get(token_out)

        if not token_in_address or not token_out_address:
            raise ValueError(
                f"Token not supported: "
                f"{token_in if not token_in_address else token_out}"
            )

        # In a real implementation, this would build the actual transaction
        # using the SUI SDK
        print(
            f"Building swap transaction: {amount_in} {token_in} -> "
            f"{min_amount_out} {token_out}"
        )

        # Return a placeholder transaction bytes
        return (
            "AQAAAAAA"
            "AgABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAABAAAAAAAAAA="
        )

    async def swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        slippage_pct: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute a swap on Cetus DEX.

        Args:
            token_in: Token to swap from (e.g., "SUI")
            token_out: Token to swap to (e.g., "USDC")
            amount_in: Amount of token_in to swap
            min_amount_out: Minimum amount of token_out to receive
            slippage_pct: Maximum acceptable slippage percentage

        Returns:
            Transaction result
        """
        # Build swap transaction
        tx_bytes = await self.build_swap_transaction(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
        )

        # Execute transaction
        result = await self.sui_client.execute_transaction(tx_bytes)
        return result

    async def get_liquidity_positions(self) -> List[Dict[str, Any]]:
        """Get all liquidity positions for the current address."""
        # Fetch all objects owned by the address
        objects = await self.sui_client.get_objects()

        # Filter for liquidity position objects
        positions = []
        for obj in objects:
            if (
                "type" in obj
                and "cetus" in obj["type"].lower()
                and "position" in obj["type"].lower()
            ):
                position_data = await self.sui_client.get_object(
                    obj["objectId"]
                )
                positions.append(position_data)

        return positions
