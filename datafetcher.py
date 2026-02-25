"""
Data acquisition module for fetching market data from multiple sources.
Handles errors, rate limiting, and data validation.
"""
import asyncio
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class DataFetchError(Exception):
    """Custom exception for data fetching failures"""
    pass

class MarketDataFetcher:
    """Multi-source market data fetcher with fallback strategies"""
    
    def __init__(self, exchange_id: str = "binance", sandbox: bool = True):
        """
        Initialize data fetcher with exchange configuration.
        
        Args:
            exchange_id: Exchange identifier (binance, coinbase, etc.)
            sandbox: Whether to use sandbox/testnet mode
        """
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self._exchange = None
        self._async_exchange = None
        self._last_fetch_time = 0
        self._min_fetch_interval = 1.0  # Rate limiting
        
        # Initialize synchronously
        self._init_sync_exchange()
    
    def _init_sync_exchange(self):
        """Initialize synchronous exchange instance"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            
            if self.sandbox:
                config['options']['sandboxMode'] = True
            
            self._exchange = exchange_class(config)
            logger.info("Initialized %s exchange (