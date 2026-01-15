"""
Anomaly detection using K-Means, DBSCAN, and rule-based methods.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Base class for anomaly detectors."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_cols = ['ret_z', 'volz', 'range_pct']
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract feature matrix from dataframe, removing NaN rows.
        
        Returns:
            X: Feature matrix (n_samples, 3)
            df_clean: DataFrame with only valid rows
        """
        # Remove rows with NaN features
        df_clean = df.dropna(subset=self.feature_cols).copy()
        
        if len(df_clean) == 0:
            return np.array([]).reshape(0, 3), df_clean
        
        X = df_clean[self.feature_cols].values
        
        return X, df_clean


class KMeansDetector(AnomalyDetector):
    """K-Means based anomaly detector with per-cluster thresholds."""
    
    def __init__(self, n_clusters: int = 5, percentile: float = 97.5):
        """
        Args:
            n_clusters: Number of clusters (tune on validation set)
            percentile: Percentile threshold per cluster (e.g., 97.5 for top 2.5%)
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.percentile = percentile
        self.kmeans = None
        self.cluster_thresholds = {}
    
    def fit(self, df_train: pd.DataFrame):
        """
        Fit K-Means and compute cluster-specific distance thresholds.
        
        Args:
            df_train: Training data with features
        """
        X_train, df_clean = self.prepare_features(df_train)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data after removing NaN")
        
        print(f"\n[K-Means] Fitting on {len(X_train)} training samples...")
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Compute silhouette score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
            print(f"  Silhouette score: {sil_score:.3f}")
        
        # Compute distances to nearest centroid
        distances = np.linalg.norm(
            X_scaled - self.kmeans.cluster_centers_[labels], 
            axis=1
        )
        
        # Compute per-cluster thresholds at specified percentile
        for cluster_id in range(self.n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_distances = distances[cluster_mask]
            
            if len(cluster_distances) > 0:
                threshold = np.percentile(cluster_distances, self.percentile)
                self.cluster_thresholds[cluster_id] = threshold
                print(f"  Cluster {cluster_id}: {cluster_mask.sum()} points, "
                      f"threshold={threshold:.3f}")
        
        self.fitted = True
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using K-Means distance thresholds.
        
        Args:
            df: DataFrame to predict on
            
        Returns:
            DataFrame with added columns: kmeans_anomaly, kmeans_distance, kmeans_cluster
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        X, df_clean = self.prepare_features(df)
        
        if len(X) == 0:
            # No valid data - return original df with default values
            df_result = df.copy()
            df_result['kmeans_anomaly'] = 0
            df_result['kmeans_distance'] = np.nan
            df_result['kmeans_cluster'] = -1
            return df_result
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        clusters = self.kmeans.predict(X_scaled)
        
        # Compute distances to assigned centroids
        distances = np.linalg.norm(
            X_scaled - self.kmeans.cluster_centers_[clusters], 
            axis=1
        )
        
        # Flag anomalies based on cluster-specific thresholds
        anomalies = np.zeros(len(distances), dtype=int)
        for i, (cluster, dist) in enumerate(zip(clusters, distances)):
            threshold = self.cluster_thresholds.get(cluster, np.inf)
            if dist > threshold:
                anomalies[i] = 1
        
        # Add results to cleaned dataframe
        df_clean['kmeans_anomaly'] = anomalies
        df_clean['kmeans_distance'] = distances
        df_clean['kmeans_cluster'] = clusters
        
        # Merge back with original dataframe
        df_result = df.merge(
            df_clean[['Date', 'ticker', 'kmeans_anomaly', 'kmeans_distance', 'kmeans_cluster']],
            on=['Date', 'ticker'],
            how='left'
        )
        
        # Fill NaN with defaults
        df_result['kmeans_anomaly'] = df_result['kmeans_anomaly'].fillna(0).astype(int)
        df_result['kmeans_cluster'] = df_result['kmeans_cluster'].fillna(-1).astype(int)
        
        return df_result


class DBSCANDetector(AnomalyDetector):
    """DBSCAN based anomaly detector (noise points = anomalies)."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        """
        Args:
            eps: Maximum distance for neighborhood (tune on validation set)
            min_samples: Minimum points to form dense region
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, df_train: pd.DataFrame):
        """
        Fit scaler on training data.
        
        Note: DBSCAN itself is not fitted - it's run fresh each time
        in walk-forward mode.
        """
        X_train, _ = self.prepare_features(df_train)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data")
        
        print(f"\n[DBSCAN] Fitting scaler on {len(X_train)} training samples...")
        print(f"  Parameters: eps={self.eps}, min_samples={self.min_samples}")
        
        # Fit scaler
        self.scaler.fit(X_train)
        self.fitted = True
    
    def predict(self, df: pd.DataFrame, refit: bool = True) -> pd.DataFrame:
        """
        Predict anomalies using DBSCAN.
        
        Walk-forward approach: Refit DBSCAN on expanding window.
        
        Args:
            df: DataFrame to predict on
            refit: If True, refit DBSCAN on this data
            
        Returns:
            DataFrame with added columns: dbscan_anomaly, dbscan_cluster
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        X, df_clean = self.prepare_features(df)
        
        if len(X) == 0:
            df_result = df.copy()
            df_result['dbscan_anomaly'] = 0
            df_result['dbscan_cluster'] = -1
            return df_result
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Points labeled -1 are noise (anomalies)
        anomalies = (clusters == -1).astype(int)
        
        # Add results
        df_clean['dbscan_anomaly'] = anomalies
        df_clean['dbscan_cluster'] = clusters
        
        # Merge back
        df_result = df.merge(
            df_clean[['Date', 'ticker', 'dbscan_anomaly', 'dbscan_cluster']],
            on=['Date', 'ticker'],
            how='left'
        )
        
        df_result['dbscan_anomaly'] = df_result['dbscan_anomaly'].fillna(0).astype(int)
        df_result['dbscan_cluster'] = df_result['dbscan_cluster'].fillna(-1).astype(int)
        
        return df_result


class RuleBasedDetector:
    """
    Simple rule-based anomaly detector (baseline).
    
    Flags anomaly if:
    - |ret_z| > 2.5, OR
    - volz > 2.5, OR
    - range_pct > 95
    """
    
    def __init__(self, 
                 ret_z_threshold: float = 2.5, 
                 volz_threshold: float = 2.5,
                 range_pct_threshold: float = 95.0):
        self.ret_z_threshold = ret_z_threshold
        self.volz_threshold = volz_threshold
        self.range_pct_threshold = range_pct_threshold
        
        print(f"\n[Rule-Based] Thresholds:")
        print(f"  |ret_z| > {ret_z_threshold}")
        print(f"  volz > {volz_threshold}")
        print(f"  range_pct > {range_pct_threshold}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using simple rules.
        
        Returns:
            DataFrame with added columns: rule_anomaly, anomaly_type
        """
        df = df.copy()
        
        # Initialize flags
        df['rule_anomaly'] = 0
        df['anomaly_type'] = ''
        
        # Check conditions (handle NaN safely)
        ret_z_extreme = df['ret_z'].abs() > self.ret_z_threshold
        volz_extreme = df['volz'] > self.volz_threshold
        range_extreme = df['range_pct'] > self.range_pct_threshold
        
        # Flag anomalies
        anomaly_mask = ret_z_extreme | volz_extreme | range_extreme
        df.loc[anomaly_mask, 'rule_anomaly'] = 1
        
        # Label anomaly types
        crash = (df['ret'] < 0) & ret_z_extreme
        spike = (df['ret'] > 0) & ret_z_extreme
        volume_shock = volz_extreme
        
        # Assign type labels
        df.loc[crash & ~volume_shock, 'anomaly_type'] = 'crash'
        df.loc[spike & ~volume_shock, 'anomaly_type'] = 'spike'
        df.loc[crash & volume_shock, 'anomaly_type'] = 'crash + volume_shock'
        df.loc[spike & volume_shock, 'anomaly_type'] = 'spike + volume_shock'
        df.loc[volume_shock & ~crash & ~spike, 'anomaly_type'] = 'volume_shock'
        
        return df


class MarketAnomalyDetector:
    """Detect market-wide anomaly days."""
    
    def __init__(self, 
                 market_ret_percentile: float = 95.0, 
                 breadth_threshold: float = 0.3):
        """
        Args:
            market_ret_percentile: Percentile threshold for |market_ret|
            breadth_threshold: Threshold for low breadth (e.g., 0.3 = 30%)
        """
        self.market_ret_percentile = market_ret_percentile
        self.breadth_threshold = breadth_threshold
        
        print(f"\n[Market Detector]:")
        print(f"  |market_ret| percentile > {market_ret_percentile}")
        print(f"  OR breadth < {breadth_threshold}")
    
    def predict(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market anomaly days.
        
        Flags if:
        - |market_ret| > 95th percentile, OR
        - breadth < 30%
        """
        market_df = market_df.copy()
        
        # Market return extreme
        ret_extreme = market_df['market_ret_abs_pct'] > self.market_ret_percentile
        
        # Breadth low
        breadth_low = market_df['breadth'] < self.breadth_threshold
        
        # Flag market anomaly
        market_df['market_anomaly'] = (ret_extreme | breadth_low).astype(int)
        
        return market_df


if __name__ == "__main__":
    from data_loader import StockDataLoader
    from features import FeatureEngine
    
    print("="*60)
    print("Anomaly Detectors - Test")
    print("="*60)
    
    # Load and prepare data
    loader = StockDataLoader("data/raw")
    tickers = ['QQQ', 'AAPL', 'MSFT']
    df = loader.load_universe(tickers)
    splits = loader.split_train_val_test(df)
    
    # Compute features
    fe = FeatureEngine()
    train = fe.compute_features(splits['train'])
    test = fe.compute_features(splits['test'])
    
    # Test K-Means detector
    print("\n" + "="*60)
    kmeans = KMeansDetector(n_clusters=5, percentile=97.5)
    kmeans.fit(train)
    test_pred = kmeans.predict(test)
    
    n_anomalies = test_pred['kmeans_anomaly'].sum()
    print(f"\nK-Means detected {n_anomalies} anomalies in test set")
    print(test_pred[test_pred['kmeans_anomaly'] == 1][
        ['Date', 'ticker', 'ret', 'ret_z', 'kmeans_distance']
    ].head())