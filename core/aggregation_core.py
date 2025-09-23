import logging
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from models.database import GlobalModel, Client, SessionLocal
from services.blob_service import blob_service
from services.aggregation_service import aggregation_service
from services.websocket_service import connection_manager
from utils.runtime_state import runtime_state
from functools import wraps
import time

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
            
            if not avg_weights or not blob_service.save_weights_to_blob(avg_weights, filename):
                logging.critical("Failed to save aggregated weights to blob")
                raise HTTPException(status_code=500, detail="Failed to save aggregated weights")
            
            aggregation_service.save_last_aggregation_timestamp(new_db, new_timestamp)
            logging.info(f"New timestamp saved to the database: {new_timestamp}")
            
            contributing_client_ids = [id for id, _ in weights_list_with_ids]
            new_model = GlobalModel(
                version=runtime_state.latest_version,
                num_clients_contributed=len(weights_list),
                client_ids=",".join(contributing_client_ids)
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