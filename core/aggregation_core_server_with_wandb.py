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

# FL Server imports
from evaluate_server_with_wandb import evaluate_model_with_wandb
from unified_fl_tracker import fl_tracker
from preprocessing_server import IoTDataPreprocessor

# Modern WandB imports
import wandb

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'evaluation', 'DATA', 'global_test.csv')
ARTIFACTS_PATH = os.path.join(BASE_DIR, '..', 'evaluation', 'artifacts')
FIXED_NUM_CLASSES = 15

def retry_db_operation(max_attempts=3, delay=2):
    """Retry database operations with exponential backoff"""
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

async def fl_server_aggregation_complete(db: Session, global_run_id=None):
    """
    COMPLETE FL Server Aggregation with comprehensive WandB tracking
    Goal: Aggregate client weights, evaluate global model, track FL progress
    """
    
    print("\n" + "="*80)
    print("🔄 FEDERATED LEARNING - SERVER AGGREGATION")
    print("="*80)
    print("Goal: Aggregate client weights and evaluate new global model")
    print("Input: Client model weights from blob storage")  
    print("Output: New global model + comprehensive FL metrics")
    print("="*80)
    
    try:
        runtime_state.update_checked_timestamp()
        last_aggregation_timestamp = aggregation_service.load_last_aggregation_timestamp(db)
        runtime_state.last_aggregation_timestamp = last_aggregation_timestamp or 0

        logging.info(f"🎯 Starting FL Server Aggregation")
        logging.info(f"Last processed timestamp: {runtime_state.last_aggregation_timestamp}")
        
        # Initialize FL Server WandB run (linked to global training)
        if not fl_tracker.server_run:
            server_config = {
                "aggregation_algorithm": "FedAvg",
                "min_clients_per_round": 2,
                "evaluation_dataset": "global_test_set", 
                "fixed_num_classes": FIXED_NUM_CLASSES,
                "artifacts_path": ARTIFACTS_PATH,
                "test_data_path": TEST_DATA_PATH,
                "linked_to_global_run": global_run_id
            }
            
            fl_tracker.initialize_server_aggregation_run(
                global_run_id=global_run_id,
                config=server_config
            )

        # Load FL model architecture
        model = blob_service.get_model_architecture()
        if not model:
            logging.critical("❌ Failed to load FL model architecture")
            raise HTTPException(status_code=500, detail="Failed to load model architecture")

        # Load client weights from blob storage
        weights_list_with_ids, num_examples_list, loss_list, new_timestamp = blob_service.load_weights_from_blob(
            runtime_state.last_aggregation_timestamp
        )

        if not weights_list_with_ids:
            logging.info("ℹ️ No new client weights found")
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}

        if len(weights_list_with_ids) < 2:
            logging.info("ℹ️ Insufficient client weights for FL aggregation")
            return {"status": "no_update", "message": "Only 1 client weight found", "num_clients": 1}

        if not num_examples_list:
            logging.error("❌ Client example counts missing")
            return {"status": "error", "message": "Example counts missing for aggregation"}

        # Extract FL round information
        contributing_client_ids = [client_id for client_id, _ in weights_list_with_ids]
        total_samples = sum(num_examples_list)
        
        print(f"\n🔗 FL Round Information:")
        print(f"   Contributing Clients: {len(weights_list_with_ids)}")
        print(f"   Client IDs: {contributing_client_ids}")
        print(f"   Total Samples: {total_samples}")
        print(f"   Sample Distribution: {dict(zip(contributing_client_ids, num_examples_list))}")
        
        # Log client participation to WandB
        participation_metrics = {
            "aggregation/num_contributing_clients": len(weights_list_with_ids),
            "aggregation/total_client_samples": total_samples,
            "aggregation/timestamp": new_timestamp,
            "aggregation/client_participation_rate": len(weights_list_with_ids) / 10  # Assuming 10 total clients
        }
        
        # Log individual client contributions
        for i, (client_id, examples) in enumerate(zip(contributing_client_ids, num_examples_list)):
            participation_metrics[f"aggregation/client_{client_id}_samples"] = examples
            participation_metrics[f"aggregation/client_{client_id}_weight"] = examples / total_samples
        
        wandb.log(participation_metrics)

        # Database operations with retry logic
        new_db = SessionLocal()
        try:
            logging.info("📊 Fetching latest FL model version")
            
            @retry_db_operation(max_attempts=3, delay=2)
            def get_latest_fl_model(db: Session):
                return db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()

            latest_model = get_latest_fl_model(new_db)
            runtime_state.latest_version = latest_model.version if latest_model else 0
            logging.info(f"Current FL model version: {runtime_state.latest_version}")
            
            runtime_state.latest_version += 1
            current_fl_round = runtime_state.latest_version

            if new_db.query(GlobalModel).filter_by(version=current_fl_round).first():
                logging.error(f"❌ Duplicate FL round detected: {current_fl_round}")
                raise HTTPException(status_code=409, detail=f"FL Round {current_fl_round} already exists")

            filename = aggregation_service.get_versioned_filename(current_fl_round)
            logging.info(f"🔄 Preparing FL Round {current_fl_round}: {filename}")

            # Perform Federated Averaging
            weights_list = [weights for _, weights in weights_list_with_ids]
            print(f"\n🧮 Federated Averaging:")
            print(f"   Aggregating weights from {len(weights_list)} clients")
            print(f"   Weighting by sample counts: {num_examples_list}")
            
            # Log aggregation details
            aggregation_details = {
                f"fl_round_{current_fl_round}/num_weight_layers": len(weights_list[0]) if weights_list else 0,
                f"fl_round_{current_fl_round}/aggregation_method": "FedAvg",
                f"fl_round_{current_fl_round}/client_weights": dict(zip(contributing_client_ids, 
                                                                       [w/total_samples for w in num_examples_list]))
            }
            wandb.log(aggregation_details)

            avg_weights = aggregation_service.federated_averaging(weights_list, num_examples_list)
            logging.info("✅ Federated averaging completed successfully")

            # Build and compile aggregated global model
            model = blob_service.get_model_architecture()
            model.set_weights(avg_weights)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logging.info("✅ FL model compiled for evaluation")

            # Prepare test data with FL server preprocessing
            print(f"\n🔧 FL Server Preprocessing:")
            print(f"   Artifacts path: {ARTIFACTS_PATH}")
            print(f"   Test data path: {TEST_DATA_PATH}")
            
            try:
                preprocessor = IoTDataPreprocessor(artifacts_path=ARTIFACTS_PATH)
            except Exception as e:
                print(f"❌ FL preprocessing failed: {e}")
                raise

            X_test, y_test, num_classes_test = preprocessor.preprocess_data(TEST_DATA_PATH)
            y_test_cat = to_categorical(y_test, num_classes=FIXED_NUM_CLASSES)

            print(f"✅ FL test data prepared:")
            print(f"   Test samples: {X_test.shape[0]}")
            print(f"   Test features: {X_test.shape[1]}")
            print(f"   Classes found: {num_classes_test}")
            print(f"   Fixed encoding: {FIXED_NUM_CLASSES} classes")

            # Get class names for FL evaluation
            try:
                class_names = preprocessor.global_le.classes_.tolist()
                print(f"✅ Class names loaded: {len(class_names)} classes")
            except AttributeError:
                class_names = [
                    'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                    'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                    'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
                ]
                print(f"⚠️ Using fallback class names: {len(class_names)} classes")

            # Comprehensive FL model evaluation
            print(f"\n📊 FL ROUND {current_fl_round} EVALUATION:")
            print("-" * 60)
            
            eval_results = evaluate_model_with_wandb(
                model, X_test, y_test_cat,
                class_names=class_names,
                fl_round=current_fl_round,
                log_to_wandb=True
            )
            test_metrics = eval_results['test']

            print(f"✅ FL Round {current_fl_round} Results:")
            print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
            print(f"   Test Loss: {test_metrics['loss']:.4f}")
            print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"   Weighted F1: {test_metrics['weighted_f1']:.4f}")

            # Log comprehensive FL metrics to WandB tracker
            fl_tracker.log_fl_round_results(
                fl_round=current_fl_round,
                test_metrics=test_metrics,
                num_contributing_clients=len(weights_list_with_ids),
                client_ids=contributing_client_ids,
                model_version=current_fl_round
            )

            # Create FL performance trends (after round 2+)
            if current_fl_round > 1:
                try:
                    fl_tracker.create_fl_performance_trends()
                    print("📈 FL performance trends updated")
                except Exception as e:
                    logging.warning(f"Could not create FL trends: {e}")

            # Save FL model with comprehensive metadata
            fl_metadata = {
                "fl_round": str(current_fl_round),
                "aggregation_method": "FedAvg",
                "contributing_clients": len(weights_list_with_ids),
                "total_samples_aggregated": str(total_samples),
                "client_ids": contributing_client_ids,
                "client_sample_distribution": dict(zip(contributing_client_ids, num_examples_list)),
                "test_accuracy": str(test_metrics['accuracy']),
                "test_loss": str(test_metrics['loss']),
                "test_macro_f1": str(test_metrics['macro_f1']),
                "test_weighted_f1": str(test_metrics['weighted_f1']),
                "test_macro_precision": str(test_metrics['macro_precision']),
                "test_macro_recall": str(test_metrics['macro_recall']),
                "evaluation_timestamp": datetime.now().isoformat(),
                "wandb_server_run_id": wandb.run.id if wandb.run else "none",
                "wandb_server_url": wandb.run.url if wandb.run else "none"
            }

            if not avg_weights or not blob_service.save_weights_to_blob(avg_weights, filename, fl_metadata):
                logging.critical("❌ Failed to save FL aggregated weights")
                raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

            # Save aggregation timestamp
            aggregation_service.save_last_aggregation_timestamp(new_db, new_timestamp)
            logging.info(f"✅ FL timestamp saved: {new_timestamp}")

            # Update database with FL round information
            new_fl_model = GlobalModel(
                version=current_fl_round,
                test_metrics=test_metrics,
                num_clients_contributed=len(weights_list),
                client_ids=contributing_client_ids
            )

            new_db.add(new_fl_model)
            new_db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
                {"contribution_count": Client.contribution_count + 1},
                synchronize_session=False
            )
            new_db.commit()
            logging.info(f"✅ FL Round {current_fl_round} saved to database")

            # Broadcast new FL model to clients
            await connection_manager.broadcast_model_update(f"NEW_FL_MODEL:{filename}")
            logging.info(f"📡 Clients notified of FL Round {current_fl_round}")

            # Log FL round success to WandB
            success_metrics = {
                f"fl_success/round_{current_fl_round}": 1,
                f"fl_success/model_version": current_fl_round,
                f"fl_success/aggregation_timestamp": new_timestamp,
                f"fl_success/server_tracking_active": True
            }
            wandb.log(success_metrics)

            print(f"\n🎉 FL ROUND {current_fl_round} COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"   Accuracy: {test_metrics['accuracy']*100:.2f}%")
            print(f"   Loss: {test_metrics['loss']:.4f}")
            print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"   Contributing Clients: {len(weights_list_with_ids)}")
            print(f"   WandB Server Run: {wandb.run.url if wandb.run else 'N/A'}")
            print("="*60)

            return {
                "status": "success",
                "message": f"FL Round {current_fl_round} completed with full WandB tracking",
                "fl_round": current_fl_round,
                "num_clients": len(weights_list),
                "test_accuracy": test_metrics['accuracy'],
                "test_loss": test_metrics['loss'],
                "test_macro_f1": test_metrics['macro_f1'],
                "contributing_clients": contributing_client_ids,
                "wandb_server_url": wandb.run.url if wandb.run else None,
                "model_filename": filename
            }

        except SQLAlchemyError as db_error:
            logging.error(f"❌ FL database error: {db_error}")
            new_db.rollback()
            if wandb.run:
                wandb.log({"fl_error/database_error": str(db_error)})
            raise HTTPException(status_code=500, detail="FL database error")
            
        finally:
            new_db.close()

    except HTTPException as http_exc:
        logging.error(f"❌ FL HTTP error: {http_exc.detail}")
        db.rollback()
        if wandb.run:
            wandb.log({"fl_error/http_error": str(http_exc.detail)})
        raise
        
    except Exception as e:
        logging.exception(f"❌ FL unexpected error: {e}")
        db.rollback()
        if wandb.run:
            wandb.log({"fl_error/unexpected_error": str(e)})
        raise HTTPException(status_code=500, detail="FL server error occurred")

# Export for usage
aggregate_weights_core = fl_server_aggregation_complete
