"""
Monthly mini-report generator.
Creates summary reports of anomalies for each month.
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import sys


class MonthlyReport:
    """Generate monthly anomaly summary reports."""
    
    def __init__(self, results_path: str = "outputs/full_results.csv"):
        """
        Initialize monthly report generator.
        
        Args:
            results_path: Path to full results CSV
        """
        self.results_path = Path(results_path)
        
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_path}\n"
                f"Please run: python -m src.walkforward first"
            )
        
        print(f"Loading results from: {self.results_path}")
        self.df = pd.read_csv(self.results_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        print(f"✓ Loaded {len(self.df):,} records")
    
    def generate_report(self, year_month: str, output_dir: str = "outputs/monthly_reports"):
        """
        Generate monthly report.
        
        Args:
            year_month: Format YYYY-MM (e.g., "2020-02")
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse year-month
        try:
            year, month = map(int, year_month.split('-'))
        except:
            print(f"❌ Invalid format: {year_month}")
            print("Please use YYYY-MM format (e.g., 2020-02)")
            return
        
        # Filter data for this month
        month_data = self.df[
            (self.df['Date'].dt.year == year) & 
            (self.df['Date'].dt.month == month)
        ].copy()
        
        if len(month_data) == 0:
            print(f"\n❌ No data found for {year_month}")
            self._print_available_months()
            return
        
        # Generate report table
        report_df = self._create_report_table(month_data)
        
        # Save report
        output_file = output_dir / f"report_{year_month}.csv"
        if len(report_df) > 0:
            report_df.to_csv(output_file, index=False)
        
        # Print report
        self._print_report(year_month, month_data, report_df, output_file)
        
        return report_df
    
    def _create_report_table(self, month_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create the monthly report table with required columns.
        
        Returns:
            DataFrame with columns: Date, Ticker, Type, ret_z, volz, Why, Mkt_flag
        """
        # Get only anomalous records
        anomalies = month_data[month_data['consensus_anomaly'] == 1].copy()
        
        if len(anomalies) == 0:
            return pd.DataFrame()
        
        # Create report
        report = []
        
        for _, row in anomalies.iterrows():
            # Create 'Why' explanation
            why_parts = []
            if pd.notna(row['ret_z']) and abs(row['ret_z']) > 2.5:
                why_parts.append(f"ret_z={row['ret_z']:.1f}")
            if pd.notna(row['volz']) and row['volz'] > 2.5:
                why_parts.append(f"volz={row['volz']:.1f}")
            if pd.notna(row['range_pct']) and row['range_pct'] > 95:
                why_parts.append(f"range_pct={row['range_pct']:.1f}")
            
            why_str = ", ".join(why_parts) if why_parts else "ML flagged"
            
            report.append({
                'Date': row['Date'].date(),
                'Ticker': row['ticker'],
                'Type': row['anomaly_type'] if pd.notna(row['anomaly_type']) else 'detected',
                'ret_z': row['ret_z'] if pd.notna(row['ret_z']) else 0,
                'volz': row['volz'] if pd.notna(row['volz']) else 0,
                'range_pct': row['range_pct'] if pd.notna(row['range_pct']) else 0,
                'Why': why_str,
                'Mkt_flag': int(row['market_anomaly']) if pd.notna(row['market_anomaly']) else 0,
                'Severity': row['severity'] if pd.notna(row['severity']) else 0,
                'Return': row['ret'] if pd.notna(row['ret']) else 0
            })
        
        report_df = pd.DataFrame(report)
        
        # Sort by date and severity
        report_df = report_df.sort_values(['Date', 'Severity'], ascending=[True, False])
        
        return report_df
    
    def _print_report(self, year_month: str, month_data: pd.DataFrame, 
                     report_df: pd.DataFrame, output_file: Path):
        """Print beautifully formatted monthly report."""
        
        month_name = datetime.strptime(year_month, '%Y-%m').strftime('%B %Y')
        
        print("\n" + "="*100)
        print(f"  MONTHLY ANOMALY REPORT: {month_name}")
        print("="*100)
        
        # ==================== SUMMARY ====================
        total_days = month_data['Date'].nunique()
        total_stock_days = len(month_data)
        anomaly_days = month_data[month_data['consensus_anomaly'] == 1]['Date'].nunique()
        total_anomalies = len(report_df)
        market_anomaly_days = month_data[month_data['market_anomaly'] == 1]['Date'].nunique()
        
        print(f"\n📈 SUMMARY")
        print("─"*100)
        print(f"  Trading days:              {total_days}")
        print(f"  Total stock-days:          {total_stock_days:,}")
        print(f"  Days with anomalies:       {anomaly_days} ({anomaly_days/total_days*100:.1f}%)")
        print(f"  Total anomalies detected:  {total_anomalies} ({total_anomalies/total_stock_days*100:.2f}%)")
        print(f"  Market anomaly days:       {market_anomaly_days} ({market_anomaly_days/total_days*100:.1f}%)")
        
        # ==================== ANOMALY BREAKDOWN ====================
        if len(report_df) > 0:
            print(f"\n📊 ANOMALY BREAKDOWN BY TYPE")
            print("─"*100)
            
            type_counts = report_df['Type'].value_counts()
            for anom_type, count in type_counts.items():
                pct = count / total_anomalies * 100
                print(f"  {anom_type:<30} {count:4d} ({pct:5.1f}%)")
        
        # ==================== TOP ANOMALIES ====================
        if len(report_df) > 0:
            print(f"\n🔥 TOP 10 ANOMALIES BY SEVERITY")
            print("─"*100)
            print(f"  {'Date':<12} {'Ticker':<8} {'Type':<22} {'Return':>8} {'ret_z':>8} "
                  f"{'volz':>8} {'Severity':>9} {'Mkt':>3}")
            print("  " + "─"*98)
            
            top_10 = report_df.nlargest(10, 'Severity')
            for _, row in top_10.iterrows():
                mkt_flag = '🚨' if row['Mkt_flag'] == 1 else '✓'
                atype = row['Type']
                if len(atype) > 22:
                    atype = atype[:19] + "..."
                    
                print(f"  {str(row['Date']):<12} {row['Ticker']:<8} {atype:<22} "
                      f"{row['Return']:>+7.2%} {row['ret_z']:>8.2f} {row['volz']:>8.2f} "
                      f"{row['Severity']:>9.1f} {mkt_flag:>3}")
        
        # ==================== MARKET STRESS DAYS ====================
        market_anomalies = month_data[month_data['market_anomaly'] == 1][
            ['Date', 'market_ret', 'breadth', 'flag_rate']
        ].drop_duplicates().sort_values('Date')
        
        if len(market_anomalies) > 0:
            print(f"\n🚨 MARKET STRESS DAYS")
            print("─"*100)
            print(f"  {'Date':<12} {'Market Ret':>12} {'Breadth':>10} {'Stocks Flagged':>15}")
            print("  " + "─"*98)
            
            for _, row in market_anomalies.iterrows():
                print(f"  {row['Date'].date():<12} {row['market_ret']:>+11.2%} "
                      f"{row['breadth']:>9.1%} {row['flag_rate']:>14.1%}")
        else:
            print(f"\n✅ No market stress days detected in {month_name}")
        
        # ==================== DAILY ANOMALY COUNTS ====================
        print(f"\n📅 DAILY ANOMALY COUNTS")
        print("─"*100)
        
        if len(report_df) > 0:
            daily_counts = report_df.groupby('Date').size().sort_values(ascending=False)
            
            print(f"  Days with most anomalies:")
            for date, count in daily_counts.head(5).items():
                # Check if market anomaly
                is_mkt = month_data[month_data['Date'].dt.date == date]['market_anomaly'].iloc[0]
                mkt_str = "🚨" if is_mkt == 1 else "  "
                print(f"    {mkt_str} {date}: {count} stocks")
        else:
            print("  No anomalies detected this month")
        
        # ==================== FILE OUTPUT ====================
        if len(report_df) > 0:
            print(f"\n💾 SAVED TO FILE")
            print("─"*100)
            print(f"  {output_file}")
            print(f"  Columns: Date, Ticker, Type, ret_z, volz, Why, Mkt_flag")
        else:
            print(f"\n  ℹ️  No anomalies detected - report not saved to file")
        
        print("\n" + "="*100 + "\n")
    
    def generate_all_months(self, output_dir: str = "outputs/monthly_reports"):
        """Generate reports for all available months."""
        # Get unique year-months
        self.df['year_month'] = self.df['Date'].dt.to_period('M')
        year_months = sorted(self.df['year_month'].unique())
        
        print(f"\n{'='*60}")
        print(f"Generating reports for {len(year_months)} months")
        print(f"{'='*60}\n")
        
        for ym in year_months:
            year_month_str = str(ym)
            print(f"Processing {year_month_str}...")
            self.generate_report(year_month_str, output_dir)
        
        print(f"\n{'='*60}")
        print(f"✓ All reports generated in: {output_dir}/")
        print(f"{'='*60}\n")
    
    def _print_available_months(self):
        """Print list of available months."""
        self.df['year_month'] = self.df['Date'].dt.to_period('M')
        months = sorted(self.df['year_month'].unique())
        
        print("\nAvailable months:")
        for ym in months:
            print(f"  {ym}")


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate monthly anomaly report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.monthly --month 2020-02
  python -m src.monthly --month 2019-12
  python -m src.monthly --all
        """
    )
    
    parser.add_argument(
        '--month',
        type=str,
        help='Month to report in YYYY-MM format (e.g., 2020-02)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate reports for all available months'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default='outputs/full_results.csv',
        help='Path to full results CSV (default: outputs/full_results.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/monthly_reports',
        help='Directory to save reports (default: outputs/monthly_reports)'
    )
    
    args = parser.parse_args()
    
    try:
        reporter = MonthlyReport(args.results)
        
        if args.all:
            reporter.generate_all_months(args.output_dir)
        elif args.month:
            reporter.generate_report(args.month, args.output_dir)
        else:
            print("❌ Error: Please specify --month YYYY-MM or --all")
            print("\nExamples:")
            print("  python -m src.monthly --month 2020-02")
            print("  python -m src.monthly --all")
            sys.exit(1)
    
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