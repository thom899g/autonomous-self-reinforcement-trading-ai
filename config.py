"""
Configuration management for the autonomous trading system.
All sensitive credentials are loaded from environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Configuration for cryptocurrency exchanges"""
    exchange_id: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox_mode: bool = True
    rate_limit: bool = True
    
    def __post_init__(self):
        """Load credentials from environment variables if not provided"""
        if not self.api_key:
            self.api_key = os.getenv("EXCHANGE_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv("EXCHANGE_API_SECRET", "")

@dataclass
class FirebaseConfig:
    """Configuration for Firebase/Firestore"""
    project_id: str = "autonomous-trading-ai"
    collection_name: str = "trading_sessions"
    
    @property
    def credentials_path(self) -> Path:
        """Path to Firebase service account credentials"""
        cred_path = Path("firebase-credentials.json")
        if not cred_path.exists():
            logger.warning("Firebase credentials file not found at %s", cred_path)
        return cred_path

@dataclass
class ModelConfig:
    """Configuration for RL model"""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    
    # Model paths
    model_save_path: Path = Path("models/ppo_trading")
    tensorboard_log: Path = Path("logs/tensorboard")

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    transaction_cost: float = 0.001  # 0.1% per trade
    lookback_window: int = 100  # Historical candles for features
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    max_daily_loss: float = 0.03  # 3%

class Config:
    """Main configuration aggregator"""
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.firebase = FirebaseConfig()
        self.model = ModelConfig()
        self.trading = TradingConfig()
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.model.model_save_path,
            self.model.tensorboard_log,
            Path("logs/system"),
            Path("data/raw"),
            Path("data/processed")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info("Ensured directory exists: %s", directory)

# Global configuration instance
config = Config()