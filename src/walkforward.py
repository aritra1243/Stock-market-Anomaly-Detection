"""
Main walk-forward anomaly detection pipeline.
Runs the full end-to-end workflow:
1. Load data
2. Compute features
3. Fit detectors on training set
4. Detect anomalies on all periods
5. Save outputs
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .data_loader import StockDataLoader
from .features import FeatureEngine
from .detectors import (KMeansDetector, DBSCANDetector, RuleBasedDetector, 
                        MarketAnomalyDetector)


class AnomalyPipeline:
    """Complete anomaly detection pipeline."""
    
    def __init__(self, 
                 tickers: list,
                 data_dir: str = "data/raw",
                 output_dir: str = "outputs",
                 # Detector hyperparameters
                 kmeans_k: int = 5,
                 kmeans_percentile: float = 97.5,
                 dbscan_eps: float = 0.5,
                 dbscan_min_samples: int = 10):
        """
        Initialize pipeline.
        
        Args:
            tickers: List of ticker symbols to analyze
            data_dir: Directory containing stock data
            output_dir: Directory to save results
            kmeans_k: Number of clusters for K-Means
            kmeans_percentile: Percentile threshold for K-Means
            dbscan_eps: Epsilon for DBSCAN
            dbscan_min_samples: Min samples for DBSCAN
        """
        self.tickers = tickers
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = StockDataLoader(data_dir)
        self.feature_engine = FeatureEngine()
        
        # Initialize detectors with specified hyperparameters
        self.kmeans = KMeansDetector(n_clusters=kmeans_k, percentile=kmeans_percentile)
        self.dbscan = DBSCANDetector(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.rule_based = RuleBasedDetector()
        self.market_detector = MarketAnomalyDetector()
        
        self.results = None
    
    def run(self):
        """Run the complete pipeline."""
        print("\n" + "="*80)
        print("STOCK MARKET ANOMALY DETECTION PIPELINE")
        print("="*80)
        print(f"\nTickers: {', '.join(self.tickers)}")
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load data
        print("\n" + "="*80)
        print("[1/6] LOADING DATA")
        print("="*80)
        df = self.loader.load_universe(self.tickers)
        
        # Step 2: Compute features on FULL dataset first (to preserve lookback history)
        print("\n" + "="*80)
        print("[2/6] COMPUTING FEATURES")
        print("="*80)
        
        # Compute features on ALL data to preserve rolling window history
        all_features = self.feature_engine.compute_features(df)
        all_market = self.feature_engine.compute_market_features(all_features)
        
        # Now split into train/val/test AFTER computing features
        train_features = all_features[(all_features['Date'] >= '2018-01-01') & 
                                       (all_features['Date'] <= '2018-12-31')].copy()
        val_features = all_features[(all_features['Date'] >= '2019-01-01') & 
                                     (all_features['Date'] <= '2019-12-31')].copy()
        test_features = all_features[(all_features['Date'] >= '2020-01-01') & 
                                      (all_features['Date'] <= '2020-03-31')].copy()
        
        # Split market features similarly
        train_market = all_market[(all_market['Date'] >= '2018-01-01') & 
                                   (all_market['Date'] <= '2018-12-31')].copy()
        val_market = all_market[(all_market['Date'] >= '2019-01-01') & 
                                 (all_market['Date'] <= '2019-12-31')].copy()
        test_market = all_market[(all_market['Date'] >= '2020-01-01') & 
                                  (all_market['Date'] <= '2020-03-31')].copy()
        
        print(f"\nData splits after feature computation:")
        print(f"  TRAIN: {len(train_features)} rows")
        print(f"  VAL:   {len(val_features)} rows")
        print(f"  TEST:  {len(test_features)} rows")

        
        # Step 3: Fit detectors on training data
        print("\n" + "="*80)
        print("[3/6] FITTING DETECTORS ON TRAINING DATA (2018)")
        print("="*80)
        self.kmeans.fit(train_features)
        self.dbscan.fit(train_features)
        
        # Step 4: Validate on validation set
        print("\n" + "="*80)
        print("[4/6] VALIDATING ON 2019 DATA")
        print("="*80)
        val_results = self._detect_anomalies(val_features, val_market)
        self._print_stats(val_results, "Validation (2019)")
        
        # Step 5: Test on test set
        print("\n" + "="*80)
        print("[5/6] TESTING ON 2020-Q1 DATA")
        print("="*80)
        test_results = self._detect_anomalies(test_features, test_market)
        self._print_stats(test_results, "Test (2020-Q1)")
        
        # Combine all results for full timeline
        train_results = self._detect_anomalies(train_features, train_market)
        all_results = pd.concat([train_results, val_results, test_results], 
                                ignore_index=True)
        all_results = all_results.sort_values(['Date', 'ticker']).reset_index(drop=True)
        
        self.results = all_results
        
        # Step 6: Save outputs
        print("\n" + "="*80)
        print("[6/6] SAVING OUTPUTS")
        print("="*80)
        self._save_outputs(all_results)
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\n✓ Results saved to: {self.output_dir}/")
        print(f"  - anomaly_cards.csv")
        print(f"  - market_days.csv")
        print(f"  - full_results.csv")
        print(f"\nNext steps:")
        print(f"  1. Query a date: python -m src.query --date 2020-02-27")
        print(f"  2. Generate monthly report: python -m src.monthly --month 2020-02")
        print("="*80 + "\n")
    
    def _detect_anomalies(self, df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all detectors on data.
        
        Returns:
            DataFrame with all detection results
        """
        # Run all detectors
        df = self.kmeans.predict(df)
        df = self.dbscan.predict(df)
        df = self.rule_based.predict(df)
        
        # Consensus: flag if ANY detector flags (union for high recall)
        df['consensus_anomaly'] = (
            (df['kmeans_anomaly'] == 1) | 
            (df['dbscan_anomaly'] == 1) | 
            (df['rule_anomaly'] == 1)
        ).astype(int)
        
        # Compute severity score (0-100)
        df['severity'] = self._compute_severity(df)
        
        # Add market-level anomalies
        market_df = self.market_detector.predict(market_df)
        df = df.merge(
            market_df[['Date', 'market_ret', 'breadth', 'market_anomaly']],
            on='Date',
            how='left'
        )
        
        # Compute flag rate per day (fraction of stocks flagged)
        flag_rate = df.groupby('Date')['consensus_anomaly'].mean().reset_index()
        flag_rate.columns = ['Date', 'flag_rate']
        df = df.merge(flag_rate, on='Date', how='left')
        
        return df
    
    def _compute_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute severity score (0-100) as average percentile of features.
        
        Higher score = more extreme across multiple dimensions.
        """
        # Convert each feature to percentile
        ret_z_pct = df['ret_z'].abs().rank(pct=True) * 100
        volz_pct = df['volz'].rank(pct=True) * 100
        range_pct = df['range_pct']  # Already in percentile form
        
        # Average across features
        severity = (ret_z_pct + volz_pct + range_pct) / 3
        
        return severity.fillna(0)
    
    def _print_stats(self, df: pd.DataFrame, split_name: str):
        """Print statistics for a data split."""
        total = len(df)
        
        if total == 0:
            print(f"\n{split_name}: No data")
            return
        
        # Count anomalies from each detector
        kmeans_flags = df['kmeans_anomaly'].sum()
        dbscan_flags = df['dbscan_anomaly'].sum()
        rule_flags = df['rule_anomaly'].sum()
        consensus_flags = df['consensus_anomaly'].sum()
        
        print(f"\n{split_name} Statistics:")
        print(f"{'='*60}")
        print(f"  Total stock-days: {total:,}")
        print(f"  K-Means anomalies: {kmeans_flags:4d} ({kmeans_flags/total*100:5.2f}%)")
        print(f"  DBSCAN anomalies:  {dbscan_flags:4d} ({dbscan_flags/total*100:5.2f}%)")
        print(f"  Rule-based:        {rule_flags:4d} ({rule_flags/total*100:5.2f}%)")
        print(f"  {'─'*60}")
        print(f"  CONSENSUS:         {consensus_flags:4d} ({consensus_flags/total*100:5.2f}%)")
        
        # Market anomalies
        unique_dates = df['Date'].nunique()
        market_anomalies = df.groupby('Date')['market_anomaly'].first().sum()
        print(f"\n  Market anomaly days: {market_anomalies} / {unique_dates} "
              f"({market_anomalies/unique_dates*100:.1f}%)")
        
        # Anomaly type breakdown
        if consensus_flags > 0:
            print(f"\n  Anomaly type breakdown:")
            type_counts = df[df['consensus_anomaly'] == 1]['anomaly_type'].value_counts()
            for atype, count in type_counts.head(5).items():
                if atype:  # Skip empty strings
                    print(f"    {atype}: {count}")
    
    def _save_outputs(self, df: pd.DataFrame):
        """Save all required outputs."""
        
        # 1. Daily Anomaly Card (only anomalies)
        anomaly_df = df[df['consensus_anomaly'] == 1].copy()
        
        if len(anomaly_df) > 0:
            # Create 'why' column explaining the anomaly
            anomaly_df['why'] = anomaly_df.apply(self._create_why_column, axis=1)
            
            anomaly_card = anomaly_df[[
                'Date', 'ticker', 'consensus_anomaly', 'anomaly_type', 
                'ret', 'ret_z', 'volz', 'range_pct', 'why'
            ]].copy()
            
            anomaly_card = anomaly_card.rename(columns={'consensus_anomaly': 'anomaly_flag'})
            anomaly_card['Date'] = anomaly_card['Date'].dt.date
            
            output_file = self.output_dir / "anomaly_cards.csv"
            anomaly_card.to_csv(output_file, index=False)
            print(f"  ✓ Saved: anomaly_cards.csv ({len(anomaly_card):,} anomalies)")
        else:
            print(f"  ⚠ No anomalies detected - anomaly_cards.csv not created")
        
        # 2. Market-Day Table
        market_df = df[['Date', 'market_ret', 'breadth', 'market_anomaly']].drop_duplicates()
        market_df = market_df.rename(columns={'market_anomaly': 'market_anomaly_flag'})
        market_df['Date'] = market_df['Date'].dt.date
        market_df = market_df.sort_values('Date')
        
        output_file = self.output_dir / "market_days.csv"
        market_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: market_days.csv ({len(market_df):,} days)")
        
        # 3. Full results (for query and analysis tools)
        output_file = self.output_dir / "full_results.csv"
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: full_results.csv ({len(df):,} records)")
    
    @staticmethod
    def _create_why_column(row) -> str:
        """Create explanation string for why this is an anomaly."""
        reasons = []
        
        if pd.notna(row['ret_z']) and abs(row['ret_z']) > 2.5:
            reasons.append(f"|ret_z|={abs(row['ret_z']):.1f}>2.5")
        
        if pd.notna(row['volz']) and row['volz'] > 2.5:
            reasons.append(f"volz={row['volz']:.1f}>2.5")
        
        if pd.notna(row['range_pct']) and row['range_pct'] > 95:
            reasons.append(f"range_pct={row['range_pct']:.1f}>95")
        
        if not reasons:
            # Flagged by ML detector but not by rules
            reasons.append("ML detector flagged")
        
        return "; ".join(reasons)


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Market Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.walkforward --universe QQQ,AAPL,MSFT,NVDA,AMZN,META
  python -m src.walkforward --universe QQQ,AAPL,MSFT --kmeans-k 7
        """
    )
    
    parser.add_argument(
        '--universe',
        type=str,
        default='QQQ,AAPL,MSFT,NVDA,AMZN,META',
        help='Comma-separated list of ticker symbols (default: QQQ,AAPL,MSFT,NVDA,AMZN,META)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing stock data (default: data/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save outputs (default: outputs)'
    )
    
    parser.add_argument(
        '--kmeans-k',
        type=int,
        default=5,
        help='Number of K-Means clusters (default: 5)'
    )
    
    parser.add_argument(
        '--kmeans-percentile',
        type=float,
        default=97.5,
        help='K-Means percentile threshold (default: 97.5)'
    )
    
    parser.add_argument(
        '--dbscan-eps',
        type=float,
        default=0.5,
        help='DBSCAN epsilon (default: 0.5)'
    )
    
    parser.add_argument(
        '--dbscan-min-samples',
        type=int,
        default=10,
        help='DBSCAN min_samples (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.universe.split(',')]
    
    try:
        # Create and run pipeline
        pipeline = AnomalyPipeline(
            tickers=tickers,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            kmeans_k=args.kmeans_k,
            kmeans_percentile=args.kmeans_percentile,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples
        )
        
        pipeline.run()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download the dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset")
        print("  2. Download and extract to: data/raw/")
        print("  3. Ensure structure is: data/raw/stocks/AAPL.csv, etc.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()