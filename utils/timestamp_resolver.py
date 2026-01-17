"""
Unified Timestamp Resolution for MLflow Artifact Tracking
Ensures consistent timestamps across data, train, and mlflow pipelines
"""

import boto3
import logging
from datetime import datetime
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import get_s3_bucket, get_aws_region
from s3_io import list_keys

logger = logging.getLogger(__name__)


class TimestampResolver:
    """
    Resolves unified timestamps across all ML pipeline artifacts.
    Ensures data, train, and mlflow artifacts share the same timestamp for lineage tracking.
    """
    
    def __init__(self):
        self.bucket = get_s3_bucket()
        self.timestamp_format = "%Y%m%d%H%M%S"
        
    def get_latest_timestamp(self, prefix: str) -> Optional[str]:
        """
        Get the latest timestamp folder from S3 prefix.
        
        Args:
            prefix: S3 prefix to search (e.g., "artifacts/data/", "artifacts/train/")
            
        Returns:
            Latest timestamp string or None if no timestamps found
        """
        try:
            all_keys = list_keys(prefix=prefix)
            
            # Extract timestamp folders from keys
            timestamps = set()
            for key in all_keys:
                # Remove prefix and extract timestamp folder
                relative_path = key[len(prefix):] if key.startswith(prefix) else key
                if '/' in relative_path:
                    timestamp_candidate = relative_path.split('/')[0]
                    # Validate timestamp format (14 digits: YYYYMMDDHHMMSS)
                    if timestamp_candidate.isdigit() and len(timestamp_candidate) == 14:
                        timestamps.add(timestamp_candidate)
            
            if timestamps:
                latest = sorted(timestamps)[-1]
                logger.info(f"📅 Found latest timestamp in {prefix}: {latest}")
                return latest
            else:
                logger.warning(f"⚠️ No valid timestamps found in {prefix}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get latest timestamp from {prefix}: {e}")
            return None
    
    def resolve_run_timestamp(self, force_new: bool = False) -> str:
        """
        Resolve unified timestamp for current pipeline run.
        
        Resolution logic (following mlflow_single_instruction_for_timestamp_alignment.md):
        1. If data pipeline artifacts exist, reuse latest data timestamp
        2. Else if train pipeline artifacts exist, reuse latest train timestamp
        3. Else create new timestamp
        
        Args:
            force_new: If True, always create new timestamp (useful for fresh runs)
            
        Returns:
            Unified timestamp string (YYYYMMDDHHMMSS)
        """
        if force_new:
            new_ts = datetime.now().strftime(self.timestamp_format)
            logger.info(f"🆕 Force new timestamp: {new_ts}")
            return new_ts
        
        # Check for latest data artifacts first (highest priority)
        latest_data_ts = self.get_latest_timestamp("artifacts/data/")
        if latest_data_ts:
            logger.info(f"🔁 Reusing latest DATA timestamp: {latest_data_ts}")
            return latest_data_ts
        
        # Check for latest train artifacts as fallback
        latest_train_ts = self.get_latest_timestamp("artifacts/train/")
        if latest_train_ts:
            logger.info(f"🔁 Reusing latest TRAIN timestamp: {latest_train_ts}")
            return latest_train_ts
        
        # No existing artifacts - create new timestamp
        new_ts = datetime.now().strftime(self.timestamp_format)
        logger.info(f"🆕 No existing timestamp found — creating new: {new_ts}")
        return new_ts
    
    def get_latest_data_timestamp(self) -> Optional[str]:
        """Get latest timestamp from data artifacts."""
        return self.get_latest_timestamp("artifacts/data/")
    
    def get_latest_train_timestamp(self) -> Optional[str]:
        """Get latest timestamp from train artifacts."""
        return self.get_latest_timestamp("artifacts/train/")
    
    def get_latest_mlflow_timestamp(self) -> Optional[str]:
        """Get latest timestamp from mlflow artifacts."""
        return self.get_latest_timestamp("artifacts/mlflow/")
    
    def validate_timestamp_sync(self) -> dict:
        """
        Validate that data, train, and mlflow artifacts are in sync.
        
        Returns:
            Dictionary with sync status and timestamps
        """
        data_ts = self.get_latest_data_timestamp()
        train_ts = self.get_latest_train_timestamp()
        mlflow_ts = self.get_latest_mlflow_timestamp()
        
        all_timestamps = {data_ts, train_ts, mlflow_ts} - {None}
        
        sync_status = {
            'data_timestamp': data_ts,
            'train_timestamp': train_ts,
            'mlflow_timestamp': mlflow_ts,
            'in_sync': len(all_timestamps) <= 1,  # All same or missing
            'latest_unified': max(all_timestamps) if all_timestamps else None
        }
        
        if sync_status['in_sync']:
            logger.info("✅ All artifacts are in sync")
        else:
            logger.warning(f"⚠️ Artifacts out of sync: data={data_ts}, train={train_ts}, mlflow={mlflow_ts}")
        
        return sync_status


def resolve_run_timestamp(force_new: bool = False) -> str:
    """
    Convenience function to resolve unified timestamp.
    
    Args:
        force_new: If True, always create new timestamp
        
    Returns:
        Unified timestamp string
    """
    resolver = TimestampResolver()
    return resolver.resolve_run_timestamp(force_new=force_new)


def get_latest_data_timestamp() -> Optional[str]:
    """Get latest data artifact timestamp."""
    resolver = TimestampResolver()
    return resolver.get_latest_data_timestamp()


def get_latest_train_timestamp() -> Optional[str]:
    """Get latest train artifact timestamp."""
    resolver = TimestampResolver()
    return resolver.get_latest_train_timestamp()


if __name__ == "__main__":
    # Test timestamp resolution
    logging.basicConfig(level=logging.INFO)
    
    resolver = TimestampResolver()
    
    print("\n" + "="*80)
    print("TIMESTAMP RESOLVER TEST")
    print("="*80)
    
    # Get latest timestamps
    print("\n📅 Latest Timestamps:")
    print(f"  • Data: {resolver.get_latest_data_timestamp()}")
    print(f"  • Train: {resolver.get_latest_train_timestamp()}")
    print(f"  • MLflow: {resolver.get_latest_mlflow_timestamp()}")
    
    # Resolve unified timestamp
    print("\n🔍 Unified Timestamp Resolution:")
    unified_ts = resolver.resolve_run_timestamp()
    print(f"  • Resolved: {unified_ts}")
    
    # Validate sync
    print("\n✅ Sync Validation:")
    sync_status = resolver.validate_timestamp_sync()
    for key, value in sync_status.items():
        print(f"  • {key}: {value}")
    
    print("\n" + "="*80)

