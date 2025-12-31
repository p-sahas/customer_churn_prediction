"""
Tests for S3 I/O utilities using moto for mocking
"""

import pytest
import pandas as pd
import json
import os
from unittest.mock import patch
from moto import mock_s3
import boto3

# Set up test environment variables
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['S3_BUCKET'] = 'test-bucket'
os.environ['S3_KMS_KEY_ARN'] = 'arn:aws:kms:us-east-1:123456789012:key/test-key'
os.environ['FORCE_S3_IO'] = 'true'

from utils.s3_io import (
    get_s3_client, put_bytes, get_bytes, write_df_csv, read_df_csv,
    write_df_json, read_df_json, write_pickle, read_pickle,
    list_keys, delete_key, key_exists
)


@pytest.fixture
def s3_setup():
    """Set up mocked S3 environment"""
    with mock_s3():
        # Create test bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        yield s3_client


class TestS3IO:
    """Test S3 I/O functionality"""
    
    def test_put_get_bytes_roundtrip(self, s3_setup):
        """Test bytes upload and download"""
        test_data = b"Hello, S3 World!"
        key = "test/bytes_test.txt"
        
        # Upload
        put_bytes(test_data, key=key, content_type="text/plain")
        
        # Download
        retrieved_data = get_bytes(key)
        
        assert retrieved_data == test_data
    
    def test_write_read_df_csv_roundtrip(self, s3_setup):
        """Test DataFrame CSV upload and download"""
        # Create test DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        key = "test/dataframe_test.csv"
        
        # Upload
        write_df_csv(df, key=key)
        
        # Download
        retrieved_df = read_df_csv(key=key)
        
        pd.testing.assert_frame_equal(df, retrieved_df)
    
    def test_write_read_df_json_roundtrip(self, s3_setup):
        """Test DataFrame JSON upload and download"""
        # Create test DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        key = "test/dataframe_test.json"
        
        # Upload
        write_df_json(df, key=key)
        
        # Download
        retrieved_df = read_df_json(key=key)
        
        pd.testing.assert_frame_equal(df, retrieved_df)
    
    def test_write_read_pickle_roundtrip(self, s3_setup):
        """Test pickle upload and download"""
        test_obj = {
            'model_params': {'n_estimators': 100, 'max_depth': 10},
            'features': ['feature1', 'feature2', 'feature3'],
            'metadata': {'created_at': '2025-10-04', 'version': '1.0'}
        }
        
        key = "test/pickle_test.pkl"
        
        # Upload
        write_pickle(test_obj, key=key)
        
        # Download
        retrieved_obj = read_pickle(key=key)
        
        assert retrieved_obj == test_obj
    
    def test_list_keys(self, s3_setup):
        """Test S3 key listing with pagination"""
        # Upload multiple test files
        test_keys = [
            "data/artifacts/csv/20251004100000/X_train.csv",
            "data/artifacts/csv/20251004100000/X_test.csv",
            "data/artifacts/csv/20251004110000/X_train.csv",
            "data/artifacts/parquet/20251004100000/X_train.parquet"
        ]
        
        for key in test_keys:
            put_bytes(b"test data", key=key)
        
        # List all keys
        all_keys = list_keys()
        assert len(all_keys) >= len(test_keys)
        
        # List with prefix
        csv_keys = list_keys(prefix="data/artifacts/csv/")
        assert len(csv_keys) == 3  # 3 CSV files
        
        parquet_keys = list_keys(prefix="data/artifacts/parquet/")
        assert len(parquet_keys) == 1  # 1 Parquet file
    
    def test_key_exists(self, s3_setup):
        """Test S3 key existence check"""
        key = "test/exists_test.txt"
        
        # Key should not exist initially
        assert not key_exists(key)
        
        # Upload file
        put_bytes(b"test data", key=key)
        
        # Key should exist now
        assert key_exists(key)
    
    def test_delete_key(self, s3_setup):
        """Test S3 key deletion"""
        key = "test/delete_test.txt"
        
        # Upload file
        put_bytes(b"test data", key=key)
        assert key_exists(key)
        
        # Delete file
        delete_key(key)
        assert not key_exists(key)
    
    def test_large_file_upload(self, s3_setup):
        """Test multipart upload for large files"""
        # Create large test data (>5MB to trigger multipart)
        large_data = b"x" * (6 * 1024 * 1024)  # 6MB
        key = "test/large_file_test.bin"
        
        # Upload
        put_bytes(large_data, key=key)
        
        # Download and verify
        retrieved_data = get_bytes(key)
        assert len(retrieved_data) == len(large_data)
        assert retrieved_data[:1000] == large_data[:1000]  # Check first 1KB
    
    def test_error_handling(self, s3_setup):
        """Test error handling for non-existent keys"""
        # Try to read non-existent key
        with pytest.raises(Exception):
            get_bytes("non/existent/key.txt")
        
        # Try to read non-existent CSV
        with pytest.raises(Exception):
            read_df_csv(key="non/existent/data.csv")
        
        # Try to read non-existent pickle
        with pytest.raises(Exception):
            read_pickle(key="non/existent/model.pkl")


class TestS3ArtifactManager:
    """Test S3 artifact management functionality"""
    
    def test_s3_artifact_paths_creation(self, s3_setup):
        """Test S3 artifact path generation"""
        from utils.s3_artifact_manager import S3ArtifactManager
        
        manager = S3ArtifactManager()
        base_names = ['X_train', 'X_test', 'Y_train', 'Y_test']
        
        # Create paths with custom timestamp
        paths = manager.create_s3_paths(base_names, timestamp='20251004120000', format_ext='csv')
        
        expected_keys = [
            'data/artifacts/csv/20251004120000/X_train.csv',
            'data/artifacts/csv/20251004120000/X_test.csv',
            'data/artifacts/csv/20251004120000/Y_train.csv',
            'data/artifacts/csv/20251004120000/Y_test.csv'
        ]
        
        for name in base_names:
            assert paths[name] in expected_keys
    
    def test_latest_artifacts_detection(self, s3_setup):
        """Test finding latest artifacts in S3"""
        from utils.s3_artifact_manager import S3ArtifactManager
        
        manager = S3ArtifactManager()
        
        # Upload test artifacts with different timestamps
        test_artifacts = [
            ('20251004100000', 'X_train.csv'),
            ('20251004110000', 'X_train.csv'),
            ('20251004120000', 'X_train.csv'),  # Latest
        ]
        
        for timestamp, filename in test_artifacts:
            key = f"data/artifacts/csv/{timestamp}/{filename}"
            put_bytes(b"test data", key=key)
        
        # Get latest artifacts
        latest_paths = manager.get_latest_artifacts(['X_train'], format_ext='csv')
        
        assert 'X_train' in latest_paths
        assert '20251004120000' in latest_paths['X_train']  # Should pick the latest timestamp
