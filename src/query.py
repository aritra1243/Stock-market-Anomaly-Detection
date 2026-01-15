"""
Date query tool: Query anomaly status for a specific date.
Input: date (YYYY-MM-DD)
Output: Market status + list of anomalous tickers
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import sys


class DateQuery:
    """Query anomaly results by date."""
    
    def __init__(self, results_path: str = "outputs/full_results.csv"):
        """
        Initialize query tool.
        
        Args:
            results_path: Path to full results CSV from walkforward.py
        """
        self.results_path = Path(results_path)
        
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_path}\n"
                f"Please run: python -m src.walkforward first"
            )
        
        # Load results
        print(f"Loading results from: {self.results_path}")
        self.df = pd.read_csv(self.results_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"✓ Loaded {len(self.df):,} records")
        print(f"  Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"  Tickers: {', '.join(sorted(self.df['ticker'].unique()))}")
    
    def query(self, date: str):
        """
        Query a specific date and print results.
        
        Args:
            date: Date string in YYYY-MM-DD format
        """
        try:
            query_date = pd.to_datetime(date)
        except:
            print(f"\n❌ Invalid date format: {date}")
            print("Please use YYYY-MM-DD format (e.g., 2020-02-27)")
            return
        
        # Filter data for this date
        day_data = self.df[self.df['Date'] == query_date]
        
        if len(day_data) == 0:
            print(f"\n❌ No data found for {date}")
            print(f"\nAvailable date range:")
            print(f"  {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
            
            # Suggest nearby dates
            all_dates = sorted(self.df['Date'].unique())
            closest = min(all_dates, key=lambda d: abs((d - query_date).days))
            print(f"\nClosest available date: {closest.date()}")
            return
        
        # Print formatted results
        self._print_query_results(query_date, day_data)
    
    def _print_query_results(self, query_date, day_data):
        """Print beautifully formatted query results."""
        
        print("\n" + "="*90)
        print(f"  ANOMALY REPORT FOR {query_date.strftime('%A, %B %d, %Y')}")
        print("="*90)
        
        # ==================== MARKET STATUS ====================
        market_ret = day_data['market_ret'].iloc[0]
        breadth = day_data['breadth'].iloc[0]
        market_anomaly = day_data['market_anomaly'].iloc[0]
        flag_rate = day_data['flag_rate'].iloc[0]
        
        print("\n📊 MARKET STATUS")
        print("─"*90)
        print(f"  Market Return:       {market_ret:+7.2%}")
        print(f"  Breadth:             {breadth:7.1%} of stocks closed positive")
        print(f"  Stocks Flagged:      {flag_rate:7.1%} ({int(flag_rate * len(day_data))} out of {len(day_data)})")
        
        if market_anomaly == 1:
            print(f"  Status:              🚨 MARKET ANOMALY DETECTED")
            
            # Explain why - check if market_ret_abs_pct column exists, otherwise use market_ret
            if 'market_ret_abs_pct' in day_data.columns:
                market_ret_abs_pct = day_data['market_ret_abs_pct'].iloc[0]
                if pd.notna(market_ret_abs_pct) and market_ret_abs_pct > 95:
                    print(f"                       → Extreme market move ({market_ret_abs_pct:.0f}th percentile)")
            elif abs(market_ret) > 0.02:  # More than 2% move
                print(f"                       → Extreme market move ({market_ret:+.2%})")
            if breadth < 0.3:
                print(f"                       → Low breadth ({breadth:.1%} < 30%)")
        else:
            print(f"  Status:              ✅ Normal market day")
        
        # ==================== ANOMALOUS STOCKS ====================
        anomalies = day_data[day_data['consensus_anomaly'] == 1].sort_values('severity', ascending=False)
        
        print(f"\n🔍 ANOMALOUS STOCKS")
        print("─"*90)
        
        if len(anomalies) == 0:
            print("  ✅ No anomalous stocks detected on this date")
        else:
            print(f"  {len(anomalies)} anomalies detected:\n")
            
            # Table header
            print(f"  {'Ticker':<8} {'Type':<22} {'Return':>8} {'ret_z':>8} {'volz':>8} "
                  f"{'range%':>8} {'Severity':>8}")
            print("  " + "─"*86)
            
            # Print each anomaly
            for _, row in anomalies.iterrows():
                ticker = row['ticker']
                atype = row['anomaly_type'] if pd.notna(row['anomaly_type']) else 'detected'
                ret = row['ret']
                ret_z = row['ret_z'] if pd.notna(row['ret_z']) else 0
                volz = row['volz'] if pd.notna(row['volz']) else 0
                range_pct = row['range_pct'] if pd.notna(row['range_pct']) else 0
                severity = row['severity'] if pd.notna(row['severity']) else 0
                
                # Truncate type if too long
                if len(atype) > 22:
                    atype = atype[:19] + "..."
                
                print(f"  {ticker:<8} {atype:<22} {ret:>+7.2%} {ret_z:>8.2f} {volz:>8.2f} "
                      f"{range_pct:>7.1f}% {severity:>8.1f}")
            
            # ==================== TOP ANOMALIES ====================
            print(f"\n  🔥 TOP 3 BY SEVERITY:")
            print("  " + "─"*86)
            
            top_3 = anomalies.head(3)
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                atype = row['anomaly_type'] if pd.notna(row['anomaly_type']) else 'detected'
                print(f"    {i}. {row['ticker']} ({atype})")
                print(f"       Return: {row['ret']:+.2%} | ret_z: {row['ret_z']:.2f} | "
                      f"volz: {row['volz']:.2f} | Severity: {row['severity']:.1f}")
            
            # ==================== DETECTOR BREAKDOWN ====================
            print(f"\n  📈 DETECTOR BREAKDOWN:")
            print("  " + "─"*86)
            
            kmeans_count = day_data['kmeans_anomaly'].sum()
            dbscan_count = day_data['dbscan_anomaly'].sum()
            rule_count = day_data['rule_anomaly'].sum()
            
            print(f"    K-Means:     {kmeans_count:3d} stocks")
            print(f"    DBSCAN:      {dbscan_count:3d} stocks")
            print(f"    Rule-based:  {rule_count:3d} stocks")
            print(f"    Consensus:   {len(anomalies):3d} stocks (union of all methods)")
        
        print("\n" + "="*90 + "\n")
    
    def query_date_range(self, start_date: str, end_date: str):
        """
        Query a date range and print summary.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return
        
        # Filter date range
        range_data = self.df[(self.df['Date'] >= start) & (self.df['Date'] <= end)]
        
        if len(range_data) == 0:
            print(f"\n❌ No data found between {start_date} and {end_date}")
            return
        
        print(f"\n{'='*80}")
        print(f"ANOMALY SUMMARY: {start_date} to {end_date}")
        print(f"{'='*80}")
        
        # Summary statistics
        n_days = range_data['Date'].nunique()
        n_market_anomalies = range_data.groupby('Date')['market_anomaly'].first().sum()
        n_stock_anomalies = range_data['consensus_anomaly'].sum()
        
        print(f"\n  Trading days: {n_days}")
        print(f"  Market anomaly days: {n_market_anomalies}")
        print(f"  Stock anomalies: {n_stock_anomalies}")
        
        # Top anomaly days
        print(f"\n  Top 5 days by stocks flagged:")
        top_days = range_data.groupby('Date')['consensus_anomaly'].sum().nlargest(5)
        for date, count in top_days.items():
            mkt_flag = range_data[range_data['Date'] == date]['market_anomaly'].iloc[0]
            mkt_str = "🚨" if mkt_flag == 1 else "  "
            print(f"    {mkt_str} {date.date()}: {int(count)} stocks")
        
        print(f"\n{'='*80}\n")


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query anomalies for a specific date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.query --date 2020-02-27
  python -m src.query --date 2020-03-16
  python -m src.query --date 2019-12-31
        """
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Date to query in YYYY-MM-DD format (e.g., 2020-02-27)'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default='outputs/full_results.csv',
        help='Path to full results CSV (default: outputs/full_results.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        query_tool = DateQuery(args.results)
        query_tool.query(args.date)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()