import logging
import os
from datetime import datetime
from fastapi import HTTPException
from tensorflow.keras.utils import to_categorical
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from db.database import GlobalModel, Client, SessionLocal
from services.blob_service import blob_service
from services.aggregation_service import aggregation_service
from services.websocket_service import connection_manager
from utils.runtime_state import runtime_state
from functools import wraps
import time
import json

from evaluation.preprocessing_server import IoTDataPreprocessor
from evaluation.evaluate import evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'evaluation', 'DATA', 'global_test.csv')

TEST_DATA_PATH = csv_path

ARTIFACTS_PATH = os.path.join(BASE_DIR, '..', 'evaluation', 'artifacts')
FIXED_NUM_CLASSES = 15

def retry_db_operation(max_attempts=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if attempt < max_attempts - 1:
                        logging.warning(f"DB operation failed, retrying {attempt + 1}/{max_attempts}: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise
        return wrapper
    return decorator

async def aggregate_weights_core(db: Session):
    try:
        runtime_state.update_checked_timestamp()
        last_aggregation_timestamp = aggregation_service.load_last_aggregation_timestamp(db)
        runtime_state.last_aggregation_timestamp = last_aggregation_timestamp or 0
        logging.info(f"Loaded last processed timestamp: {runtime_state.last_aggregation_timestamp}")

        # Initial processing steps
        model = blob_service.get_model_architecture()
        if not model:
            logging.critical("Failed to load model architecture")
            raise HTTPException(status_code=500, detail="Failed to load model architecture")
        
        weights_list_with_ids, num_examples_list, loss_list, new_timestamp = blob_service.load_weights_from_blob(
            runtime_state.last_aggregation_timestamp
        )
        
        if not weights_list_with_ids:
            logging.info("No new weights found in the blob")
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}
        
        if len(weights_list_with_ids) < 2:
            logging.info("Insufficient weights for aggregation")
            return {"status": "no_update", "message": "Only 1 weight file found", "num_clients": 1}
        
        if not num_examples_list:
            logging.error("Example counts are missing")
            return {"status": "error", "message": "Example counts missing for aggregation"}
        
        # Create a new session for database operations
        new_db = SessionLocal()
        try:
            logging.info("Fetching latest model version from database")
            
            @retry_db_operation(max_attempts=3, delay=2)
            def get_latest_model(db: Session):
                return db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
            
            latest_model = get_latest_model(new_db)
            runtime_state.latest_version = latest_model.version if latest_model else 0
            logging.info(f"Latest model version loaded: {runtime_state.latest_version}")
            
            runtime_state.latest_version += 1
            if new_db.query(GlobalModel).filter_by(version=runtime_state.latest_version).first():
                logging.error(f"Duplicate model version detected: {runtime_state.latest_version}")
                raise HTTPException(status_code=409, detail=f"Model with version {runtime_state.latest_version} already exists")
            
            filename = aggregation_service.get_versioned_filename(runtime_state.latest_version)
            logging.info(f"Preparing to save aggregated weights as: {filename}")
            
            weights_list = [weights for _, weights in weights_list_with_ids]
            logging.info(f"Aggregating weights from {len(weights_list)} clients")
            logging.info(f"Weight shapes: {[len(weights) for weights in weights_list]}")
            
            avg_weights = aggregation_service.federated_averaging(
                weights_list, num_examples_list
            )
            logging.info("Aggregation completed successfully.")

            ######################### BUILD NEW MODEL #############################
            model = blob_service.get_model_architecture()
            model.set_weights(avg_weights)

            # Model compilation is necessary to evaluate the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logging.info("Model compiled successfully for evaluation")

            logging.info("Prepare for test...")

            logging.info("🔧 Initializing preprocessor with server artifacts...")
            try:
                preprocessor = IoTDataPreprocessor(artifacts_path=ARTIFACTS_PATH)
            except Exception as e:
                logging.error(f"❌ Preprocessing initialization failed: {e}")
                logging.error("💡 Required files: preprocessor.pkl, global_label_encoder.pkl, feature_info.pkl")
                raise

            X_test, y_test, num_classes_test = preprocessor.preprocess_data(
                TEST_DATA_PATH,
            )
            y_test_cat = to_categorical(y_test, num_classes=FIXED_NUM_CLASSES)

            # FIXED: Get class names correctly from server's global label encoder
            try:
                class_names = preprocessor.global_le.classes_.tolist()  # ✅ Correct attribute
                logging.info(f"✅ Class names loaded: {class_names}")
            except AttributeError:
                # Fallback to hardcoded class names
                class_names = [
                    'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                    'Fingerlogging.infoing', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                    'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
                ]
                logging.error(f"⚠️ Using fallback class names: {len(class_names)} classes")
            
            eval_results = evaluate_model(model, X_test, y_test_cat, class_names=class_names)
            test_metrics = eval_results['test']

            metadata = {
                "final_test_loss": str(test_metrics['loss']),
                "final_test_accuracy": str(test_metrics['accuracy']),
                "final_test_precision": str(test_metrics['macro_precision']),
                "final_test_recall": str(test_metrics['macro_recall']),
                "final_test_f1": str(test_metrics['macro_f1']),
                "num_training_samples": None,
            }

            complete_metadata = {
                "test_metrics": test_metrics,
                "num_training_samples": None,
                "data_classes_present": None,
                "batch_size": None,
                "learning_rate": None,
                "differential_privacy": False,
                "noise_multiplier": None,
                "final_epsilon": None,
                "delta": None
            }

            if not avg_weights or not blob_service.save_weights_to_blob(avg_weights, filename, metadata):
                logging.critical("Failed to save aggregated weights to blob")
                raise HTTPException(status_code=500, detail="Failed to save aggregated weights")
            
            metadata_filename = aggregation_service.get_versioned_metadata_filename(runtime_state.latest_version)
            if not blob_service.upload_json_to_blob(complete_metadata, metadata_filename, {}):
                logging.critical("Failed to save metadata file to blob")
                raise HTTPException(status_code=500, detail="Failed to save metadata file")
            
            aggregation_service.save_last_aggregation_timestamp(new_db, new_timestamp)
            logging.info(f"New timestamp saved to the database: {new_timestamp}")
            
            contributing_client_ids = [id for id, _ in weights_list_with_ids]
            new_model = GlobalModel(
                version=runtime_state.latest_version,
                test_metrics=test_metrics,
                num_clients_contributed=len(weights_list),
                client_ids=contributing_client_ids
            )
            new_db.add(new_model)
            
            new_db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
                {"contribution_count": Client.contribution_count + 1},
                synchronize_session=False
            )
            new_db.commit()
            logging.info(f"Model version {runtime_state.latest_version} saved and database updated")
            
            await connection_manager.broadcast_model_update(f"NEW_MODEL:{filename}")
            logging.info(f"Clients notified of new model: {filename}")
            
            return {
                "status": "success",
                "message": f"Aggregated weights saved as {filename}",
                "num_clients": len(weights_list)
            }
        except SQLAlchemyError as db_error:
            logging.error(f"Database error during aggregation: {db_error}")
            new_db.rollback()
            raise HTTPException(status_code=500, detail="Database error occurred")
        finally:
            new_db.close()  # Ensure the new session is closed
    except HTTPException as http_exc:
        logging.error(f"HTTP Exception: {http_exc.detail}")
        db.rollback()
        raise
    except Exception as e:
        logging.exception("Unexpected error during aggregation")
        db.rollback()
        raise HTTPException(status_code=500, detail="An unexpected error occurred")