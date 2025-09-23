import logging
import tempfile
import os
import pickle
import re
from typing import List, Optional, Tuple, Any, Dict
from azure.storage.blob import BlobServiceClient, ContainerClient
from tensorflow import keras
from config.settings import settings

class BlobService:
    def __init__(self):
        try:
            self.client_blob_service = BlobServiceClient(account_url=settings.CLIENT_ACCOUNT_URL)
            self.server_blob_service = BlobServiceClient(account_url=settings.SERVER_ACCOUNT_URL)
            logging.info("Azure Blob Service clients initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Azure Blob Service: {e}")
            raise
    
    def get_model_architecture(self) -> Optional[object]:
        temp_path = None
        try:
            container_client = self.client_blob_service.get_container_client(settings.CLIENT_CONTAINER_NAME)
            blob_client = container_client.get_blob_client(settings.ARCH_BLOB_NAME)
            arch_data = blob_client.download_blob().readall()
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_file.write(arch_data)
                temp_path = temp_file.name
            
            model = keras.models.load_model(temp_path, compile=False)
            model.summary()
            return model
        except Exception as e:
            logging.error(f"Error loading model architecture: {e}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def load_weights_from_blob(self, last_aggregation_timestamp: int) -> Optional[Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[int], List[float], int]]:
        temp_path = None
        try:
            pattern = re.compile(r"localweights/client([0-9a-fA-F\-]+)_v\d+_(\d{8}_\d{6})\.pkl")
            container_client = ContainerClient.from_container_url(
                settings.LOCAL_CONTAINER_URL, 
                credential=settings.CLIENT_CONTAINER_SAS_TOKEN
            )
            
            weights_list = []
            num_examples_list = []
            loss_list = []
            new_last_aggregation_timestamp = last_aggregation_timestamp
            
            blobs = list(container_client.list_blobs())
            for blob in blobs:
                logging.info(f"Processing blob: {blob.name}")
                match = pattern.match(blob.name)
                if match:
                    client_id = match.group(1)
                    timestamp_str = match.group(2)
                    timestamp_int = int(timestamp_str.replace("_", ""))
                    
                    if timestamp_int > last_aggregation_timestamp:
                        blob_client = container_client.get_blob_client(blob.name)
                        
                        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                            download_stream = blob_client.download_blob()
                            temp_file.write(download_stream.readall())
                            temp_path = temp_file.name
                        
                        with open(temp_path, "rb") as f:
                            weights = pickle.load(f)
                        
                        blob_metadata = blob_client.get_blob_properties().metadata
                        if blob_metadata:
                            num_examples = int(blob_metadata.get('num_examples', 0))
                            loss = float(blob_metadata.get('loss', 0.0))
                            if num_examples == 0:
                                continue
                            num_examples_list.append(num_examples)
                            loss_list.append(loss)
                        
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
                            temp_path = None
                        
                        weights_list.append((client_id, weights))
                        new_last_aggregation_timestamp = max(new_last_aggregation_timestamp, timestamp_int)
                else:
                    logging.warning(f"Blob name does not match pattern: {blob.name}")
            
            if not weights_list:
                logging.info(f"No new weights found since {last_aggregation_timestamp}.")
                return None, [], [], last_aggregation_timestamp
            
            logging.info(f"Loaded weights from {len(weights_list)} files.")
            return weights_list, num_examples_list, loss_list, new_last_aggregation_timestamp
        
        except Exception as e:
            logging.error(f"Error loading weights: {e}")
            return None, [], [], last_aggregation_timestamp
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def save_weights_to_blob(self, weights: List[List[Dict[str, Any]]], filename: str) -> bool:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                temp_path = temp_file.name
                with open(temp_path, "wb") as f:
                    pickle.dump(weights, f)
            
            blob_client = self.server_blob_service.get_blob_client(
                container=settings.SERVER_CONTAINER_NAME, 
                blob=filename
            )
            with open(temp_path, "rb") as file:
                blob_client.upload_blob(file, overwrite=True)
            
            logging.info(f"Successfully saved weights to blob: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving weights to blob: {e}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

blob_service = BlobService()