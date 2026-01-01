#!/usr/bin/env python3
"""
Database Manager with RDS → SQLite Fallback
Provides unified interface for both production (RDS) and local development (SQLite)
"""

import os
import logging
import sqlite3
import psycopg2
from typing import Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Unified database manager with automatic RDS → SQLite fallback
    
    Features:
    - Try RDS first (production)
    - Fall back to SQLite if RDS fails (local development)
    - Auto-create tables in SQLite
    - Same interface for both databases
    """
    
    def __init__(
        self,
        rds_host: Optional[str] = None,
        rds_port: int = 5432,
        rds_database: Optional[str] = None,
        rds_user: Optional[str] = None,
        rds_password: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        force_sqlite: bool = False,
        use_sqlite_fallback: bool = True
    ):
        """
        Initialize database manager
        
        Args:
            rds_host: RDS hostname
            rds_port: RDS port (default: 5432)
            rds_database: RDS database name
            rds_user: RDS username
            rds_password: RDS password
            sqlite_path: Path to SQLite database file (default: ./data/local.db)
            force_sqlite: Force SQLite usage even if RDS is configured
            use_sqlite_fallback: Allow SQLite fallback if RDS fails (default: True for local dev)
        """
        self.rds_host = rds_host
        self.rds_port = rds_port
        self.rds_database = rds_database
        self.rds_user = rds_user
        self.rds_password = rds_password
        self.force_sqlite = force_sqlite
        self.use_sqlite_fallback = use_sqlite_fallback
        
        # Set SQLite path (default to ./data/local.db)
        if sqlite_path is None:
            project_root = Path(__file__).parent.parent
            sqlite_dir = project_root / "data"
            sqlite_dir.mkdir(exist_ok=True)
            self.sqlite_path = str(sqlite_dir / "local.db")
        else:
            self.sqlite_path = sqlite_path
            
        self.conn = None
        self.db_type = None  # 'rds' or 'sqlite'
        self._connect()
        
    def _connect(self):
        """Establish database connection with automatic fallback"""
        
        # If force_sqlite is True, skip RDS
        if self.force_sqlite:
            logger.info("🔧 Force SQLite mode enabled")
            self._connect_sqlite()
            return
            
        # Try RDS first
        if all([self.rds_host, self.rds_database, self.rds_user, self.rds_password]):
            try:
                logger.info(f"🔄 Attempting RDS connection: {self.rds_host}:{self.rds_port}/{self.rds_database}")
                self.conn = psycopg2.connect(
                    host=self.rds_host,
                    port=self.rds_port,
                    database=self.rds_database,
                    user=self.rds_user,
                    password=self.rds_password,
                    connect_timeout=5  # 5 second timeout
                )
                self.conn.autocommit = True
                self.db_type = 'rds'
                logger.info(f"✅ Connected to RDS: {self.rds_host}/{self.rds_database}")
                return
                
            except Exception as e:
                logger.error(f"❌ RDS connection failed: {str(e)}")
                
                # Check if SQLite fallback is allowed
                if not self.use_sqlite_fallback:
                    logger.warning("🚫 SQLite fallback is DISABLED (production mode)")
                    logger.warning("⚠️  Database operations will be SKIPPED - predictions will continue with logging only")
                    logger.info("💡 Set USE_SQLITE_FALLBACK=true to enable SQLite fallback for local development")
                    
                    # Set connection to None - graceful degradation
                    self.conn = None
                    self.db_type = 'none'
                    return
                
                logger.warning("🔄 Falling back to SQLite for local development...")
        else:
            logger.info("ℹ️  RDS credentials not configured")
            
            # Check if SQLite fallback is allowed
            if not self.use_sqlite_fallback:
                logger.warning("🚫 SQLite fallback is DISABLED (production mode)")
                logger.warning("⚠️  Database operations will be SKIPPED - predictions will continue with logging only")
                
                # Set connection to None - graceful degradation
                self.conn = None
                self.db_type = 'none'
                return
            
            logger.info("🔄 Using SQLite for local development...")
        
        # Fall back to SQLite
        self._connect_sqlite()
        
    def _connect_sqlite(self):
        """Connect to SQLite database"""
        try:
            logger.info(f"📂 Connecting to SQLite: {self.sqlite_path}")
            self.conn = sqlite3.connect(
                self.sqlite_path,
                check_same_thread=False,  # Allow multi-threaded access
                timeout=10.0
            )
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.db_type = 'sqlite'
            logger.info(f"✅ Connected to SQLite: {self.sqlite_path}")
            
            # Create tables if they don't exist
            self._create_sqlite_tables()
            
        except Exception as e:
            logger.error(f"❌ SQLite connection failed: {str(e)}")
            raise
            
    def _create_sqlite_tables(self):
        """Create SQLite tables matching RDS schema"""
        logger.info("🔧 Creating SQLite tables (if not exist)...")
        
        cursor = self.conn.cursor()
        
        # Table 1: churn_predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS churn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL,
                risk_score REAL,
                predicted_at TIMESTAMP NOT NULL,
                model_version TEXT,
                geography TEXT,
                gender TEXT,
                age INTEGER,
                tenure INTEGER,
                balance REAL,
                num_of_products INTEGER,
                has_cr_card INTEGER,
                is_active_member INTEGER,
                estimated_salary REAL,
                event_id TEXT
            )
        """)
        
        # Table 2: high_risk_customers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS high_risk_customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                geography TEXT,
                gender TEXT,
                age INTEGER,
                balance REAL,
                detected_at TIMESTAMP NOT NULL,
                UNIQUE(customer_id, detected_at)
            )
        """)
        
        # Table 3: churn_metrics_hourly
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS churn_metrics_hourly (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_timestamp TIMESTAMP NOT NULL UNIQUE,
                total_predictions INTEGER,
                churn_count INTEGER,
                churn_rate REAL,
                avg_risk_score REAL,
                high_risk_count INTEGER,
                avg_age REAL,
                avg_balance REAL,
                avg_tenure REAL
            )
        """)
        
        # Table 4: churn_metrics_daily
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS churn_metrics_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                total_predictions INTEGER,
                churn_count INTEGER,
                churn_rate REAL,
                avg_risk_score REAL,
                high_risk_count INTEGER,
                avg_age REAL,
                avg_balance REAL,
                avg_tenure REAL
            )
        """)
        
        self.conn.commit()
        logger.info("✅ SQLite tables created successfully")
        
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """
        Execute a query (INSERT, UPDATE, DELETE)
        
        Args:
            query: SQL query string
            params: Query parameters (tuple)
            
        Returns:
            Cursor object or None if no database connection
        """
        # Graceful degradation: skip if no database connection
        if self.db_type == 'none' or self.conn is None:
            logger.debug("⏭️  Skipping database operation (no connection)")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if self.db_type == 'sqlite':
                self.conn.commit()
                
            return cursor
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    def fetchone(self, query: str, params: Optional[Tuple] = None) -> Optional[Any]:
        """
        Execute a SELECT query and fetch one row
        
        Args:
            query: SQL query string
            params: Query parameters (tuple)
            
        Returns:
            Single row or None
        """
        cursor = self.execute(query, params)
        if cursor is None:
            return None
        return cursor.fetchone()
        
    def fetchall(self, query: str, params: Optional[Tuple] = None) -> list:
        """
        Execute a SELECT query and fetch all rows
        
        Args:
            query: SQL query string
            params: Query parameters (tuple)
            
        Returns:
            List of rows (empty list if no connection)
        """
        cursor = self.execute(query, params)
        if cursor is None:
            return []
        return cursor.fetchall()
        
    def is_connected(self) -> bool:
        """Check if database connection is alive"""
        try:
            if self.conn is None:
                return False
                
            if self.db_type == 'rds':
                # For PostgreSQL, check if connection is closed
                return not self.conn.closed
            else:
                # For SQLite, try a simple query
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                return True
                
        except Exception:
            return False
            
    def reconnect(self):
        """Reconnect to database"""
        logger.warning("🔄 Reconnecting to database...")
        
        # Close existing connection
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
                
        # Reconnect
        self._connect()
        
    def ensure_connection(self):
        """Ensure database connection is alive, reconnect if needed"""
        if not self.is_connected():
            self.reconnect()
            
    def close(self):
        """Close database connection"""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"✅ {self.db_type.upper()} connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
                
    def get_db_type(self) -> str:
        """Get current database type ('rds' or 'sqlite')"""
        return self.db_type
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def create_db_manager(
    rds_host: Optional[str] = None,
    rds_port: int = 5432,
    rds_database: Optional[str] = None,
    rds_user: Optional[str] = None,
    rds_password: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    force_sqlite: bool = False,
    use_sqlite_fallback: bool = True
) -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance
    
    Args:
        rds_host: RDS hostname
        rds_port: RDS port (default: 5432)
        rds_database: RDS database name
        rds_user: RDS username
        rds_password: RDS password
        sqlite_path: Path to SQLite database file
        force_sqlite: Force SQLite usage
        use_sqlite_fallback: Allow SQLite fallback if RDS fails (default: True)
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(
        rds_host=rds_host,
        rds_port=rds_port,
        rds_database=rds_database,
        rds_user=rds_user,
        rds_password=rds_password,
        sqlite_path=sqlite_path,
        force_sqlite=force_sqlite,
        use_sqlite_fallback=use_sqlite_fallback
    )

