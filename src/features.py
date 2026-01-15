"""
Feature engineering for anomaly detection.
All features use ONLY past data (no leakage) via rolling windows with .shift(1).
"""
import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """Compute rolling features for anomaly detection (leakage-free)."""
    
    def __init__(self, 
                 w_return: int = 63, 
                 w_volume: int = 21, 
                 w_range: int = 63,
                 min_obs: int = None):
        """
        Initialize feature engine with rolling window sizes.
        
        Args:
            w_return: Window for return statistics (default: 63 trading days)
            w_volume: Window for volume statistics (default: 21 days)
            w_range: Window for range statistics (default: 63 days)
            min_obs: Minimum observations before scoring (default: max of windows)
        """
        self.w_return = w_return
        self.w_volume = w_volume
        self.w_range = w_range
        self.min_obs = min_obs or max(w_return, w_volume, w_range)
        
        print(f"\nFeature Engine Config:")
        print(f"  Return window: {self.w_return} days")
        print(f"  Volume window: {self.w_volume} days")
        print(f"  Range window: {self.w_range} days")
        print(f"  Warm-up period: {self.min_obs} days")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for each ticker using only past data.
        
        Features computed:
        - ret: daily return
        - ret_z: return z-score (using past 63 days)
        - volz: log-volume z-score (using past 21 days)
        - range_pct: intraday range percentile (vs past 63 days)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        print("\nComputing features...")
        
        # Process each ticker separately
        result_dfs = []
        tickers = df['ticker'].unique()
        
        for ticker in tickers:
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = self._compute_ticker_features(ticker_df)
            result_dfs.append(ticker_df)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        result = result.sort_values(['Date', 'ticker']).reset_index(drop=True)
        
        # Report feature statistics
        n_total = len(result)
        n_valid = result['sufficient_history'].sum()
        print(f"  Total rows: {n_total}")
        print(f"  Rows with sufficient history: {n_valid} ({n_valid/n_total*100:.1f}%)")
        print(f"  Features: ret, ret_z, volz, range_pct")
        
        return result
    
    def _compute_ticker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single ticker.
        
        CRITICAL: All rolling statistics use .shift(1) to avoid leakage!
        """
        df = df.copy().sort_values('Date').reset_index(drop=True)
        
        # 1. Daily return (using Adj Close for split/dividend adjustment)
        df['ret'] = df['Adj Close'].pct_change()
        
        # 2. Log volume (add small constant to avoid log(0))
        df['log_volume'] = np.log(df['Volume'] + 1)
        
        # 3. Intraday range (normalized by close price)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
        
        # 4. Return Z-Score (PAST DATA ONLY via .shift(1))
        # Mean and std computed on [t-63, t-1], then applied to day t
        df['ret_mean'] = df['ret'].rolling(
            window=self.w_return, min_periods=1
        ).mean().shift(1)
        
        df['ret_std'] = df['ret'].rolling(
            window=self.w_return, min_periods=1
        ).std().shift(1)
        
        df['ret_z'] = (df['ret'] - df['ret_mean']) / (df['ret_std'] + 1e-8)
        
        # 5. Volume Z-Score (PAST DATA ONLY via .shift(1))
        df['vol_mean'] = df['log_volume'].rolling(
            window=self.w_volume, min_periods=1
        ).mean().shift(1)
        
        df['vol_std'] = df['log_volume'].rolling(
            window=self.w_volume, min_periods=1
        ).std().shift(1)
        
        df['volz'] = (df['log_volume'] - df['vol_mean']) / (df['vol_std'] + 1e-8)
        
        # 6. Intraday Range Percentile (PAST DATA ONLY)
        # For each day, compute percentile vs previous W_range days
        df['range_pct'] = 0.0
        
        for i in range(len(df)):
            if i < self.w_range:
                # Not enough history
                df.loc[i, 'range_pct'] = np.nan
            else:
                # Compare current range to past W_range days
                current_range = df.loc[i, 'intraday_range']
                past_ranges = df.loc[i-self.w_range:i-1, 'intraday_range'].values
                
                if len(past_ranges) > 0:
                    # Percentile rank: fraction of past values below current
                    percentile = (past_ranges < current_range).sum() / len(past_ranges) * 100
                    df.loc[i, 'range_pct'] = percentile
                else:
                    df.loc[i, 'range_pct'] = np.nan
        
        # 7. Mark rows with sufficient history
        df['sufficient_history'] = (df.index >= self.min_obs)
        
        # Set features to NaN where insufficient history
        feature_cols = ['ret_z', 'volz', 'range_pct']
        for col in feature_cols:
            df.loc[~df['sufficient_history'], col] = np.nan
        
        return df
    
    def compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market-level features for each date.
        
        Features:
        - market_ret: mean return across all tickers
        - breadth: fraction of tickers with positive returns
        - market_ret_abs_pct: percentile of |market_ret| vs past
        
        Args:
            df: DataFrame with computed stock features
            
        Returns:
            DataFrame with market-level features by date
        """
        print("\nComputing market features...")
        
        # Aggregate by date
        market_df = df.groupby('Date').agg({
            'ret': ['mean', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
        }).reset_index()
        
        market_df.columns = ['Date', 'market_ret', 'breadth']
        market_df = market_df.sort_values('Date').reset_index(drop=True)
        
        # Compute rolling stats for market return (PAST DATA ONLY)
        market_df['market_ret_mean'] = market_df['market_ret'].rolling(
            window=self.w_return, min_periods=1
        ).mean().shift(1)
        
        market_df['market_ret_std'] = market_df['market_ret'].rolling(
            window=self.w_return, min_periods=1
        ).std().shift(1)
        
        # Percentile of |market_ret| vs past
        market_df['market_ret_abs_pct'] = 0.0
        
        for i in range(len(market_df)):
            if i < self.w_return:
                market_df.loc[i, 'market_ret_abs_pct'] = np.nan
            else:
                current_abs = abs(market_df.loc[i, 'market_ret'])
                past_abs = market_df.loc[i-self.w_return:i-1, 'market_ret'].abs().values
                
                if len(past_abs) > 0:
                    pct = (past_abs < current_abs).sum() / len(past_abs) * 100
                    market_df.loc[i, 'market_ret_abs_pct'] = pct
                else:
                    market_df.loc[i, 'market_ret_abs_pct'] = np.nan
        
        print(f"  Market features computed for {len(market_df)} dates")
        
        return market_df


if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    print("="*60)
    print("Feature Engineering - Test")
    print("="*60)
    
    # Load data
    loader = StockDataLoader("data/raw")
    tickers = ['QQQ', 'AAPL', 'MSFT']
    df = loader.load_universe(tickers)
    
    # Compute features
    fe = FeatureEngine()
    df_features = fe.compute_features(df)
    market_features = fe.compute_market_features(df_features)
    
    # Display sample
    print("\nSample features (first valid rows):")
    valid_df = df_features[df_features['sufficient_history']]
    print(valid_df[['Date', 'ticker', 'ret', 'ret_z', 'volz', 'range_pct']].head(10))
    
    print("\nSample market features:")
    print(market_features[['Date', 'market_ret', 'breadth', 'market_ret_abs_pct']].head(10))