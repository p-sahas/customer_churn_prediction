#!/usr/bin/env python3
"""
Kafka Consumer with Real-time ML Inference
Micro-batch processing: 1000 samples OR 30 seconds timeout
"""

import json
import logging
import argparse
import os
import sys
import time
from typing import Dict, Any, List
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from confluent_kafka import Consumer, Producer, KafkaError
from src.model_inference import ModelInference
from utils.config import load_config
from utils.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Kafka configuration
KAFKA_CONFIG = config.get('kafka', {})
INPUT_TOPIC = KAFKA_CONFIG.get('topics', {}).get('customer_events', 'customer-events')
OUTPUT_TOPIC = KAFKA_CONFIG.get('topics', {}).get('predictions', 'churn-predictions')
BOOTSTRAP_SERVERS = KAFKA_CONFIG.get('bootstrap_servers', os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'))
BATCH_SIZE = KAFKA_CONFIG.get('consumer', {}).get('batch_size', int(os.getenv('BATCH_SIZE', 1000)))
TIMEOUT_SECONDS = KAFKA_CONFIG.get('consumer', {}).get('timeout_seconds', int(os.getenv('TIMEOUT_SECONDS', 30)))

# RDS configuration
RDS_CONFIG = config.get('rds', {})
RDS_HOST = RDS_CONFIG.get('host', os.getenv('RDS_HOST'))
RDS_PORT = RDS_CONFIG.get('port', int(os.getenv('RDS_PORT', 5432)))
RDS_DB = RDS_CONFIG.get('database', os.getenv('RDS_DB_NAME'))
RDS_USER = RDS_CONFIG.get('username', os.getenv('RDS_USERNAME'))
RDS_PASSWORD = RDS_CONFIG.get('password', os.getenv('RDS_PASSWORD'))

# SQLite fallback configuration (default: True for local dev, False for ECS production)
USE_SQLITE_FALLBACK = os.getenv('USE_SQLITE_FALLBACK', 'true').lower() in ('true', '1', 'yes')


class ChurnInferenceConsumer:
    """Real-time churn inference consumer with micro-batch processing"""
    
    def __init__(self):
        self.model = None
        self.db_manager = None
        self.producer = None
        
    def initialize(self):
        """Initialize ML model, database connection, and Kafka producer"""
        try:
            # Initialize ML model from S3/MLflow
            logger.info("=" * 60)
            logger.info("INITIALIZING CHURN INFERENCE CONSUMER")
            logger.info("=" * 60)
            
            logger.info("Loading sklearn model from S3/MLflow...")
            # Use unified timestamp structure: artifacts/train/{timestamp}/churn_model.pkl
            # ModelInference will automatically find the latest timestamp
            model_path = "artifacts/train/churn_model"  # Base path (timestamp auto-resolved)
            self.model = ModelInference(model_path=model_path, use_spark=False)
            logger.info("✅ ML model loaded successfully")
            
            # Initialize database connection (RDS with optional SQLite fallback)
            logger.info(f"🔧 SQLite fallback: {'ENABLED' if USE_SQLITE_FALLBACK else 'DISABLED (production mode)'}")
            self.db_manager = DatabaseManager(
                rds_host=RDS_HOST,
                rds_port=RDS_PORT,
                rds_database=RDS_DB,
                rds_user=RDS_USER,
                rds_password=RDS_PASSWORD,
                use_sqlite_fallback=USE_SQLITE_FALLBACK
            )
            logger.info(f"✅ Database connection established ({self.db_manager.get_db_type().upper()})")
            
            # Initialize Kafka producer for predictions
            self.producer = Producer({'bootstrap.servers': BOOTSTRAP_SERVERS})
            logger.info(f"✅ Kafka producer initialized (server: {BOOTSTRAP_SERVERS})")
            
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def extract_customer_data(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate customer data from Kafka message"""
        # Handle nested structure
        customer_data = message_data.get('data', message_data)
        
        # Required fields with defaults
        return {
            'CustomerId': customer_data.get('CustomerId', 0),
            'Geography': customer_data.get('Geography', 'Unknown'),
            'Gender': customer_data.get('Gender', 'Unknown'),
            'Age': customer_data.get('Age', 0),
            'CreditScore': customer_data.get('CreditScore', 600),
            'Balance': customer_data.get('Balance', 0.0),
            'EstimatedSalary': customer_data.get('EstimatedSalary', 0.0),
            'Tenure': customer_data.get('Tenure', 0),
            'NumOfProducts': customer_data.get('NumOfProducts', 1),
            'HasCrCard': customer_data.get('HasCrCard', 0),
            'IsActiveMember': customer_data.get('IsActiveMember', 0),
        }
    
    def write_prediction_to_rds(self, customer_id: str, customer_data: Dict, prediction_result: Dict, event_id: str = None):
        """Write individual prediction to database (RDS or SQLite)"""
        if not self.db_manager:
            return
        
        try:
            # Ensure connection is alive
            self.db_manager.ensure_connection()
            
            # Parse prediction
            status = prediction_result.get('Status', 'Unknown')
            confidence_str = prediction_result.get('Confidence', '0%')
            probability = float(confidence_str.replace('%', '')) / 100.0
            prediction_value = 1 if status == 'Churn' else 0
            risk_score = probability if prediction_value == 1 else (1 - probability)
            
            # Extract customer features
            geography = customer_data.get('Geography', 'Unknown')
            gender = customer_data.get('Gender', 'Unknown')
            age = customer_data.get('Age', 0)
            tenure = customer_data.get('Tenure', 0)
            balance = customer_data.get('Balance', 0.0)
            num_products = customer_data.get('NumOfProducts', 1)
            has_cr_card = customer_data.get('HasCrCard', 0)
            is_active = customer_data.get('IsActiveMember', 0)
            salary = customer_data.get('EstimatedSalary', 0.0)
            
            # Get model version (if available)
            model_version = "sklearn_v1.0"  # TODO: Get from model metadata
            
            # Insert prediction (handle both PostgreSQL and SQLite syntax)
            if self.db_manager.get_db_type() == 'rds':
                # PostgreSQL syntax with ON CONFLICT
                insert_query = """
                    INSERT INTO churn_predictions (
                        customer_id, prediction, probability, risk_score,
                        predicted_at, model_version, geography, gender, age, tenure,
                        balance, num_of_products, has_cr_card, is_active_member,
                        estimated_salary, event_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                """
            else:
                # SQLite syntax with INSERT OR IGNORE
                insert_query = """
                    INSERT OR IGNORE INTO churn_predictions (
                        customer_id, prediction, probability, risk_score,
                        predicted_at, model_version, geography, gender, age, tenure,
                        balance, num_of_products, has_cr_card, is_active_member,
                        estimated_salary, event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            
            self.db_manager.execute(insert_query, (
                str(customer_id), prediction_value, probability, risk_score,
                datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), model_version, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active, salary, event_id
            ))
            
        except Exception as e:
            logger.error(f"Failed to write prediction to database: {e}")
    
    def process_batch(self, max_messages: int = BATCH_SIZE, timeout: int = TIMEOUT_SECONDS, 
                     group_id: str = None) -> int:
        """Process batch of messages with ML predictions and write to RDS + Kafka"""
        
        # Configure consumer
        if group_id is None:
            group_id = f"churn-inference-{int(time.time())}"
        
        consumer_config = {
            'bootstrap.servers': BOOTSTRAP_SERVERS,
            'group.id': group_id,
            'auto.offset.reset': 'earliest' if 'batch_' in group_id else 'latest',
            'enable.auto.commit': True
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([INPUT_TOPIC])
        
        # Collect messages (micro-batch)
        messages = []
        start_time = time.time()
        
        logger.info(f"📥 Collecting messages (max: {max_messages}, timeout: {timeout}s)...")
        
        while len(messages) < max_messages and (time.time() - start_time) < timeout:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    break
                continue
            
            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                messages.append(message_data)
            except json.JSONDecodeError:
                logger.warning("Failed to decode message")
                continue
        
        consumer.close()
        
        if not messages:
            logger.info("⚠️  No messages to process")
            return 0
        
        # Process with ML
        logger.info(f"🧠 Processing {len(messages)} messages with ML inference...")
        logger.info("=" * 70)
        logger.info(f"{'Status':<8} | {'Customer':<10} | {'Location':<10} | {'Prediction':<10} | {'Confidence':<10} | {'Risk':<8}")
        logger.info("-" * 70)
        
        processed = 0
        batch_start = time.time()
        
        for i, message_data in enumerate(messages):
            try:
                # Extract customer data
                customer_data = self.extract_customer_data(message_data)
                customer_id = customer_data.get('CustomerId', 'N/A')
                geography = str(customer_data.get('Geography', 'N/A'))[:8]
                event_id = message_data.get('event_id', f"evt_{customer_id}_{int(time.time())}")
                
                # Make prediction
                prediction = self.model.predict(customer_data)
                status = prediction.get('Status', 'Unknown')
                confidence = prediction.get('Confidence', '0%')
                
                # Calculate risk score
                probability = float(confidence.replace('%', '')) / 100.0
                is_churn = (status == 'Churn')
                risk_score = probability if is_churn else (1 - probability)
                
                # Display result
                pred_emoji = "🔴" if is_churn else "🟢"
                logger.info(f"{pred_emoji:8} | {str(customer_id)[:10]:10s} | {geography:10s} | {status:10s} | {confidence:10s} | {risk_score:.4f}")
                
                # Prepare result
                result = {
                    'customer_id': str(customer_id),
                    'prediction': 1 if is_churn else 0,
                    'probability': probability,
                    'risk_score': risk_score,
                    'status': status,
                    'confidence': confidence,
                    'geography': customer_data.get('Geography'),
                    'gender': customer_data.get('Gender'),
                    'age': customer_data.get('Age'),
                    'balance': customer_data.get('Balance'),
                    'processed_at': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                    'event_id': event_id,
                    'model_version': 'sklearn_v1.0'
                }
                
                # Write to RDS
                self.write_prediction_to_rds(customer_id, customer_data, prediction, event_id)
                
                # Send to Kafka
                self.producer.produce(
                    topic=OUTPUT_TOPIC,
                    key=str(customer_id),
                    value=json.dumps(result, default=str)
                )
                
                processed += 1
                
            except Exception as e:
                logger.error(f"  ❌   | ERROR    | ERROR      | FAILED     | ERROR      | ERROR")
                logger.error(f"Error processing message {i}: {str(e)}")
        
        self.producer.flush()
        
        batch_duration = time.time() - batch_start
        
        logger.info("-" * 70)
        logger.info(f"✅ Batch completed: {processed}/{len(messages)} predictions in {batch_duration:.2f}s")
        logger.info(f"   Throughput: {processed/batch_duration:.1f} predictions/sec")
        logger.info("=" * 70)
        
        return processed
    
    def run_continuous(self, poll_interval: int = 5, show_progress: bool = True):
        """Run continuous micro-batch processing"""
        logger.info("=" * 60)
        logger.info("🔄 STARTING CONTINUOUS INFERENCE")
        logger.info(f"Batch size: {BATCH_SIZE} | Timeout: {TIMEOUT_SECONDS}s")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        total_processed = 0
        batch_count = 0
        
        try:
            while True:
                batch_count += 1
                logger.info(f"\n📡 Batch #{batch_count} - Waiting for messages...")
                
                # Process micro-batch
                processed = self.process_batch(
                    max_messages=BATCH_SIZE,
                    timeout=TIMEOUT_SECONDS,
                    group_id='churn-inference-continuous'
                )
                
                if processed > 0:
                    total_processed += processed
                    logger.info(f"✅ Batch #{batch_count} complete (Total: {total_processed} predictions)")
                else:
                    if show_progress:
                        logger.info("⏳ No new messages - waiting...")
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            logger.info(f"\n🛑 Continuous processing stopped")
            logger.info(f"Total processed: {total_processed} predictions across {batch_count} batches")
    
    def close(self):
        """Close all connections"""
        if self.db_conn:
            self.db_conn.close()
        if self.producer:
            self.producer.flush()
        logger.info("✅ Consumer shutdown complete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Kafka Consumer with Real-time ML Inference")
    parser.add_argument('--max-messages', type=int, default=BATCH_SIZE)
    parser.add_argument('--timeout', type=int, default=TIMEOUT_SECONDS)
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('--poll-interval', type=int, default=5, help='Seconds between batch polls')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 KAFKA CHURN INFERENCE CONSUMER")
        logger.info(f"Input Topic: {INPUT_TOPIC}")
        logger.info(f"Output Topic: {OUTPUT_TOPIC}")
        logger.info(f"Kafka Server: {BOOTSTRAP_SERVERS}")
        
        consumer = ChurnInferenceConsumer()
        if not consumer.initialize():
            return 1
        
        if args.continuous:
            consumer.run_continuous(args.poll_interval, not args.quiet)
        else:
            processed = consumer.process_batch(args.max_messages, args.timeout)
            return 0 if processed > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Consumer failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        if 'consumer' in locals():
            consumer.close()


if __name__ == "__main__":
    exit(main())
