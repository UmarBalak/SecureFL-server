import logging
import time
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from db.database import GlobalAggregation, GlobalModel, Client

class AggregationService:
    @staticmethod
    def load_last_aggregation_timestamp(db: Session) -> int:
        for attempt in range(3):
            try:
                timestamp = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
                return int(timestamp.value) if timestamp else 0
            except OperationalError as db_error:
                logging.error(f"Attempt {attempt + 1} - Database error: {db_error}")
                db.rollback()
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
    
    @staticmethod
    def save_last_aggregation_timestamp(db: Session, new_timestamp):
        try:
            timestamp_record = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
            if timestamp_record:
                timestamp_record.value = new_timestamp
            else:
                new_record = GlobalAggregation(key="last_aggregation_timestamp", value=new_timestamp)
                db.add(new_record)
            db.commit()
        except Exception as e:
            logging.error(f"Error saving last aggregation timestamp: {e}")
            raise
    
    @staticmethod
    def federated_averaging(
        weights_list: List[List[Any]], 
        num_examples_list: List[int]
    ) -> List[Any]:
        num_clients = len(weights_list)
        if num_clients == 0:
            logging.error("No weights provided for aggregation.")
            return None
        
        # Validate input lengths
        if len(weights_list) != len(num_examples_list):
            logging.error("Mismatched lengths: weights and examples")
            return None
        
        # Validate tensor shapes
        if weights_list:
            expected_length = len(weights_list[0])
            for weights in weights_list:
                if len(weights) != expected_length:
                    logging.error("Inconsistent tensor lengths in weights")
                    return None
        
        # Simple FedAvg: equal weights for all clients
        weights = np.array([1.0 / num_clients] * num_clients)
        logging.info(f"Performing FedAvg with {num_clients} clients, each with weight {1.0 / num_clients}")
        
        # Perform weighted average using NumPy
        avg_weights = []
        for layer_idx in range(len(weights_list[0])):
            layer_weights = np.array([client_weights[layer_idx] for client_weights in weights_list])
            weighted_sum = np.average(layer_weights, axis=0, weights=weights)
            avg_weights.append(weighted_sum)
        
        return avg_weights
    
    @staticmethod
    def get_versioned_filename(version: int, prefix="g", extension=".pkl"):
        return f"{prefix}{version}{extension}"

aggregation_service = AggregationService()