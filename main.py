import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Response, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import atexit

# Local imports
from config.settings import settings
from db.database import get_db, Client, GlobalModel, GlobalAggregation, InitialModel, SessionLocal
from services.websocket_service import connection_manager
from utils.runtime_state import runtime_state
from core.aggregation_core import aggregate_weights_core

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)

# Initialize scheduler globally
scheduler = AsyncIOScheduler()

# Database ping function - async for AsyncIOScheduler
async def ping_database():
    db = SessionLocal()
    try:
        # Use async execution
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db.execute(text("SELECT 1"))
        )
        db.commit()
        logging.debug("Database ping successful")
    except Exception as e:
        logging.error(f"Database ping failed: {e}")
    finally:
        db.close()

# Scheduled weight aggregation function - async
async def scheduled_aggregate_weights():
    logging.info("Scheduled task: Starting weight aggregation process.")
    db = SessionLocal()
    try:
        await aggregate_weights_core(db)
        logging.info("Scheduled weight aggregation completed successfully.")
    except Exception as e:
        logging.error(f"Error during scheduled weight aggregation: {e}")
    finally:
        db.close()

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting up FastAPI application...")
    
    # Add scheduled jobs
    scheduler.add_job(
        ping_database,
        IntervalTrigger(seconds=60),
        id='database_ping',
        name='Database Ping',
        replace_existing=True
    )
    
    scheduler.add_job(
        scheduled_aggregate_weights,
        CronTrigger(minute="*/2"),
        id='aggregate_weights',
        name='Aggregate Weights',
        replace_existing=True
    )
    
    # Start the scheduler
    try:
        scheduler.start()
        logging.info("Scheduler started successfully!")
        
        # Log scheduled jobs
        jobs = scheduler.get_jobs()
        logging.info(f"Active scheduled jobs: {len(jobs)}")
        for job in jobs:
            logging.info(f"Job: {job.name} - Next run: {job.next_run_time}")
            
    except Exception as e:
        logging.error(f"Failed to start scheduler: {e}")
    
    yield  # This is where the application runs
    
    # Shutdown
    logging.info("Shutting down FastAPI application...")
    try:
        scheduler.shutdown(wait=False)
        logging.info("Scheduler shutdown completed.")
    except Exception as e:
        logging.error(f"Error during scheduler shutdown: {e}")

# Create FastAPI app with lifespan
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
        global_aggregation = db.execute(select(GlobalAggregation)).scalars().all()
        initial_model = db.execute(select(InitialModel)).scalars().all()
        return {
            "clients": clients,
            "global_models": global_models,
            "global_aggregation": global_aggregation,
            "initial_model": initial_model,
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
        latest_model_version = f"g{runtime_state.latest_version}.h5"
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

# Diagnostic endpoint to check scheduler status
@app.get("/scheduler-status")
async def scheduler_status():
    if not scheduler.running:
        return {"status": "not_running", "jobs": []}
    
    jobs = scheduler.get_jobs()
    job_info = []
    for job in jobs:
        job_info.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time),
            "trigger": str(job.trigger)
        })
    
    return {
        "status": "running",
        "job_count": len(jobs),
        "jobs": job_info
    }

# Also ensure cleanup on process exit as backup
atexit.register(lambda: scheduler.shutdown(wait=False) if scheduler.running else None)