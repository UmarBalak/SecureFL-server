import logging
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Response, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Local imports
from config.settings import settings
from db.database import get_db, Client, GlobalModel, GlobalAggregation, SessionLocal
from services.websocket_service import connection_manager
from utils.runtime_state import runtime_state
from core.aggregation_core import aggregate_weights_core

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)

# FastAPI app
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    logging.info("Server starting up...")
    production_scheduler.start()
    yield
    # Shutdown  
    logging.info("Server shutting down...")
    production_scheduler.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def verify_admin(api_key: str):
    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized admin access")

@app.get("/")
async def root():
    return {"message": "HELLO, WORLD. Welcome to the SecureFL Server!"}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)

@app.get("/get_data")
async def get_all_data(db: Session = Depends(get_db)):
    try:
        clients = db.execute(select(Client)).scalars().all()
        global_models = db.execute(select(GlobalModel)).scalars().all()
        global_vars_table = db.execute(select(GlobalAggregation)).scalars().all()
        return {
            "clients": clients,
            "global_models": global_models,
            "global_aggregation": global_vars_table,
            "last_checked_timestamp": runtime_state.last_checked_timestamp
        }
    except Exception as e:
        logging.error(f"Error in /get_data endpoint: {e}")
        return {"error": "Failed to fetch data. Please try again later."}

@app.post("/register")
async def register(
    csn: str = Body(..., embed=True),
    admin_api_key: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    try:
        verify_admin(admin_api_key)
        existing_client = db.query(Client).filter(Client.csn == csn).first()
        if existing_client:
            raise HTTPException(status_code=400, detail="Client already registered")
        
        client_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        new_client = Client(csn=csn, client_id=client_id, api_key=api_key)
        db.add(new_client)
        db.commit()
        
        return {
            "status": "success",
            "message": "Client registered successfully",
            "data": {"client_id": client_id, "api_key": api_key}
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error during client registration: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the registration")

@app.get("/aggregate-weights")
async def aggregate_weights(db: Session = Depends(get_db)):
    try:
        return await aggregate_weights_core(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, db: Session = Depends(get_db)):
    retry_attempts = 3
    client = None
    
    try:
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if not client:
            logging.warning(f"Client {client_id} not found in database. Closing WebSocket.")
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        logging.info(f"Client {client_id} found in DB: {client}")
        await connection_manager.connect(client_id, websocket)
        
        # Update client status to Active
        for attempt in range(retry_attempts):
            try:
                client.status = "Active"
                db.commit()
                logging.info(f"Client {client_id} connected successfully, status updated to 'Active'.")
                break
            except SQLAlchemyError as db_error:
                db.rollback()
                logging.error(f"Attempt {attempt + 1} - Failed to update 'Active' status for {client_id}: {db_error}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    raise
        
        await websocket.send_text(f"Your status is now: {client.status}")
        
        # Send latest model version
        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        runtime_state.latest_version = latest_model.version if latest_model else 0
        latest_model_version = f"g{runtime_state.latest_version}.pkl"
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                if not data:
                    break
                
                # Ensure client status is active
                for attempt in range(retry_attempts):
                    try:
                        db.refresh(client)
                        if client.status != "Active":
                            client.status = "Active"
                            db.commit()
                            await websocket.send_text(f"Your updated status is: {client.status}")
                        break
                    except SQLAlchemyError as db_error:
                        db.rollback()
                        logging.error(f"Attempt {attempt + 1} - Database error for client {client_id}: {db_error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2)
                        else:
                            raise
            except WebSocketDisconnect:
                logging.info(f"Client {client_id} disconnected gracefully.")
                break
            except Exception as e:
                logging.error(f"Error handling message from client {client_id}: {e}")
                await websocket.send_text("An error occurred. Please try again later.")
                break
    
    except SQLAlchemyError as db_error:
        logging.error(f"Database error for client {client_id}: {db_error}")
        db.rollback()
        await websocket.close(code=1002, reason="Database error.")
    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected unexpectedly.")
    except Exception as e:
        logging.error(f"Unexpected error for client {client_id}: {e}")
        await websocket.close(code=1000, reason="Internal server error.")
    
    finally:
        # Cleanup
        for attempt in range(retry_attempts):
            try:
                await connection_manager.disconnect(client_id)
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Failed to disconnect client {client_id}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
        
        # Update client status to Inactive
        if client:
            for attempt in range(retry_attempts):
                try:
                    db.refresh(client)
                    client.status = "Inactive"
                    db.commit()
                    logging.info(f"Client {client_id} is now inactive. DB updated successfully.")
                    break
                except SQLAlchemyError as db_error:
                    db.rollback()
                    logging.error(f"Attempt {attempt + 1} - Failed to update 'Inactive' status for {client_id}: {db_error}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2)
                    else:
                        raise
        
        logging.info(f"Cleanup completed for client {client_id}.")

# Production scheduler for reliable task execution
import threading
import time
from datetime import datetime, timedelta
from sqlalchemy.sql import text

class ProductionScheduler:
    def __init__(self, interval_minutes=5, ping_interval_seconds=60):
        self.interval_minutes = interval_minutes
        self.ping_interval_seconds = ping_interval_seconds
        self.running = False
        self.thread = None
        self.ping_thread = None
        self.is_executing = False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.ping_thread.start()
        logging.info(f"Scheduler started - running every {self.interval_minutes} minutes, pinging DB every {self.ping_interval_seconds} seconds")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.ping_thread:
            self.ping_thread.join(timeout=5)
    
    def _ping_loop(self):
        while self.running:
            db = None
            for attempt in range(3):
                try:
                    db = SessionLocal()
                    db.execute(text("SELECT 1"))
                    db.commit()
                    logging.debug("Database ping successful")
                    break
                except Exception as e:
                    logging.error(f"Database ping failed (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(2)
                finally:
                    if db:
                        db.close()
            time.sleep(self.ping_interval_seconds)
    
    def _run_loop(self):
        next_run = datetime.utcnow() + timedelta(minutes=self.interval_minutes)
        while self.running:
            current_time = datetime.utcnow()
            if current_time >= next_run and not self.is_executing:
                self._execute_task()
                next_run = current_time + timedelta(minutes=self.interval_minutes)
                logging.info(f"Next run scheduled for: {next_run}")
            time.sleep(10)
    
    def _execute_task(self):
        if self.is_executing:
            return
        def run_aggregation():
            self.is_executing = True
            db = None
            try:
                logging.info("Starting scheduled aggregation...")
                db = SessionLocal()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(aggregate_weights_core(db))
                logging.info(f"Scheduled aggregation completed: {result}")
            except Exception as e:
                logging.error(f"Scheduled aggregation failed: {e}")
            finally:
                if db:
                    db.close()
                self.is_executing = False
                if 'loop' in locals():
                    loop.close()
        threading.Thread(target=run_aggregation, daemon=True).start()

# Initialize and start scheduler
production_scheduler = ProductionScheduler(interval_minutes=5, ping_interval_seconds=60)
production_scheduler.start()

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Server...")
    uvicorn.run(app, host="localhost", port=8080)