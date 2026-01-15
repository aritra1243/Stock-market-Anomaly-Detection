"""
Data loading and preprocessing for stock market data.
Handles loading individual ticker CSVs and combining into a unified dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
    """Load and preprocess stock data from Kaggle dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing the stocks/ folder with CSV files
        """
        self.data_dir = Path(data_dir)
        self.stocks_dir = self.data_dir / "stocks"
        
        if not self.stocks_dir.exists():
            raise FileNotFoundError(
                f"Stocks directory not found: {self.stocks_dir}\n"
                f"Please download data from Kaggle and extract to {self.data_dir}"
            )
    
    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Load data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.stocks_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {ticker}: {missing}")
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Basic data quality checks
        df = df[df['Volume'] > 0].copy()  # Remove zero volume days
        df = df.dropna(subset=['Adj Close']).copy()  # Remove missing prices
        
        return df
    
    def load_universe(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load data for multiple tickers and combine.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Combined DataFrame with all tickers
        """
        dfs = []
        
        print(f"\nLoading {len(tickers)} tickers...")
        for ticker in tickers:
            try:
                df = self.load_ticker(ticker)
                dfs.append(df)
                print(f"  ✓ {ticker}: {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded for any ticker")
        
        # Combine all tickers
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(['Date', 'ticker']).reset_index(drop=True)
        
        print(f"\nTotal records: {len(combined)}")
        print(f"Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
        
        return combined
    
    def filter_date_range(self, df: pd.DataFrame, start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """
        Filter dataframe by date range.
        
        Args:
            df: Input DataFrame
            start_date: Start date (inclusive) in YYYY-MM-DD format
            end_date: End date (inclusive) in YYYY-MM-DD format
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        
        if start_date:
            df_filtered = df_filtered[df_filtered['Date'] >= start_date]
        if end_date:
            df_filtered = df_filtered[df_filtered['Date'] <= end_date]
        
        return df_filtered.reset_index(drop=True)
    
    def split_train_val_test(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/val/test sets based on assignment specifications.
        
        Train: 2018
        Val: 2019
        Test: 2020 Q1 (Jan-Mar)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        splits = {
            'train': self.filter_date_range(df, '2018-01-01', '2018-12-31'),
            'val': self.filter_date_range(df, '2019-01-01', '2019-12-31'),
            'test': self.filter_date_range(df, '2020-01-01', '2020-03-31')
        }
        
        print("\nData splits:")
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                print(f"  {split_name.upper():5s}: {split_df['Date'].min().date()} to "
                      f"{split_df['Date'].max().date()} | {len(split_df):5d} rows | "
                      f"{split_df['ticker'].nunique()} tickers")
            else:
                print(f"  {split_name.upper():5s}: No data")
        
        return splits


def download_kaggle_data(data_dir: str = "data/raw"):
    """
    Download data from Kaggle using Kaggle API.
    
    Prerequisites:
        1. Install kaggle: pip install kaggle
        2. Setup API credentials: ~/.kaggle/kaggle.json
        3. Get credentials from: https://www.kaggle.com/account
    
    Args:
        data_dir: Directory to save downloaded data
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Install with: pip install kaggle"
        )
    
    dataset = "jacksoncrow/stock-market-dataset"
    output_path = Path(data_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset} to {output_path}...")
    kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)
    print("✓ Download complete!")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Stock Data Loader - Test")
    print("="*60)
    
    # Uncomment to download data:
    # download_kaggle_data()
    
    # Load sample tickers
    loader = StockDataLoader("data/raw")
    tickers = ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META']
    
    try:
        df = loader.load_universe(tickers)
        splits = loader.split_train_val_test(df)
        print("\n✓ Data loaded successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease download data first:")
        print("  1. Go to: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset")
        print("  2. Download and extract to: data/raw/")