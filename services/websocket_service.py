import logging
from typing import Dict
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, client_id: str):
        for attempt in range(3):
            try:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                    logging.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Error disconnecting client {client_id}: {e}")
                if attempt < 2:
                    import time
                    time.sleep(2)
    
    async def broadcast_model_update(self, message: str):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                logging.info(f"Update notification sent to client {client_id}")
            except Exception as e:
                logging.error(f"Failed to send update to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

connection_manager = ConnectionManager()