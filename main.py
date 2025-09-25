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
import traceback

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

# Add a logger for this module
logger = logging.getLogger(__name__)

# Local imports
logger.info("=== TESTING LOCAL IMPORTS ===")
try:
    from config.settings import settings
    from db.database import get_db, Client, GlobalModel, GlobalAggregation, InitialModel, SessionLocal
    from services.websocket_service import connection_manager
    from utils.runtime_state import runtime_state
    from core.aggregation_core import aggregate_weights_core
    logger.info("✓ Local import successful")
except Exception as e:
    logger.error(f"✗ Failed to import local modules: {e}")




# Initialize scheduler globally - but don't start it yet
scheduler = None

def init_scheduler():
    """Initialize scheduler safely"""
    global scheduler
    try:
        if scheduler is None:
            scheduler = AsyncIOScheduler()
            logger.info("=== SCHEDULER INITIALIZED ===")
        return scheduler
    except Exception as e:
        logger.error(f"=== FAILED TO INITIALIZE SCHEDULER: {e} ===")
        logger.error(traceback.format_exc())
        raise

# Database ping function with better error handling
async def ping_database():
    logger.info("=== DATABASE PING STARTED ===")
    db = None
    try:
        db = SessionLocal()
        # Simplified ping - just execute a simple query
        result = db.execute(text("SELECT 1")).fetchone()
        logger.info(f"=== DATABASE PING SUCCESSFUL: {result} ===")
        return True
    except Exception as e:
        logger.error(f"=== DATABASE PING FAILED: {e} ===")
        logger.error(traceback.format_exc())
        return False
    finally:
        if db:
            try:
                db.close()
            except Exception as e:
                logger.error(f"Error closing DB connection: {e}")

# Scheduled weight aggregation function with better error handling
async def scheduled_aggregate_weights():
    logger.info("=== SCHEDULED TASK: Starting weight aggregation process ===")
    db = None
    try:
        db = SessionLocal()
        result = await aggregate_weights_core(db)
        logger.info("=== SCHEDULED weight aggregation completed successfully ===")
        return result
    except Exception as e:
        logger.error(f"=== ERROR during scheduled weight aggregation: {e} ===")
        logger.error(traceback.format_exc())
        return None
    finally:
        if db:
            try:
                db.close()
            except Exception as e:
                logger.error(f"Error closing DB connection: {e}")

# Lifespan context manager with comprehensive error handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    
    # Startup
    logger.info("=== STARTING UP FASTAPI APPLICATION ===")
    
    try:
        # Initialize scheduler
        scheduler = init_scheduler()
        
        # Test database connectivity first
        logger.info("=== TESTING DATABASE CONNECTION ===")
        db_ok = await ping_database()
        if not db_ok:
            logger.warning("=== DATABASE CONNECTION FAILED - PROCEEDING WITHOUT SCHEDULER ===")
        else:
            logger.info("=== DATABASE CONNECTION OK ===")
            
            # Add scheduled jobs only if DB is working
            try:
                scheduler.add_job(
                    ping_database,
                    IntervalTrigger(seconds=60),
                    id='database_ping',
                    name='Database Ping',
                    replace_existing=True,
                    max_instances=1,  # Prevent overlapping executions
                    misfire_grace_time=30
                )
                logger.info("=== DATABASE PING JOB ADDED ===")
                
                scheduler.add_job(
                    scheduled_aggregate_weights,
                    CronTrigger(hour=3, minute=0),  # Runs every day at 03:00
                    id='aggregate_weights',
                    name='Aggregate Weights',
                    replace_existing=True,
                    max_instances=1,  # Prevent overlapping executions
                    misfire_grace_time=60
                )
                logger.info("=== AGGREGATE WEIGHTS JOB ADDED ===")
                
                # Start the scheduler
                if not scheduler.running:
                    scheduler.start()
                    logger.info("=== SCHEDULER STARTED SUCCESSFULLY ===")
                
                # Log scheduled jobs
                jobs = scheduler.get_jobs()
                logger.info(f"=== ACTIVE SCHEDULED JOBS: {len(jobs)} ===")
                for job in jobs:
                    logger.info(f"=== JOB: {job.name} - Next run: {job.next_run_time} ===")
                    
            except Exception as scheduler_error:
                logger.error(f"=== SCHEDULER SETUP FAILED: {scheduler_error} ===")
                logger.error(traceback.format_exc())
                logger.warning("=== CONTINUING WITHOUT SCHEDULER ===")
        
        logger.info("=== FASTAPI STARTUP COMPLETED SUCCESSFULLY ===")
        
    except Exception as startup_error:
        logger.error(f"=== CRITICAL STARTUP ERROR: {startup_error} ===")
        logger.error(traceback.format_exc())
        # Don't raise - let the app start even if scheduler fails
        logger.warning("=== CONTINUING WITH LIMITED FUNCTIONALITY ===")
    
    yield  # This is where the application runs
    
    # Shutdown
    logger.info("=== SHUTTING DOWN FASTAPI APPLICATION ===")
    try:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("=== SCHEDULER SHUTDOWN COMPLETED ===")
    except Exception as e:
        logger.error(f"=== ERROR DURING SCHEDULER SHUTDOWN: {e} ===")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="SecureFL Server", version="1.0.0")

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
    return {"message": "HELLO, WORLD. Welcome to the SecureFL Server!", "status": "running"}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    # Enhanced health check
    scheduler_status = "not_initialized"
    if scheduler:
        scheduler_status = "running" if scheduler.running else "stopped"
    
    return {
        "status": "healthy",
        "scheduler_status": scheduler_status,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)

# Test logging endpoint
@app.get("/test-logging")
async def test_logging():
    logger.info("=== LOGGING TEST - THIS SHOULD APPEAR ===")
    return {"message": "Check logs for test message", "timestamp": asyncio.get_event_loop().time()}

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
        logger.error(f"Error in /get_data endpoint: {e}")
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
        logger.error(f"Error during client registration: {e}")
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
            logger.warning(f"Client {client_id} not found in database. Closing WebSocket.")
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        logger.info(f"Client {client_id} found in DB: {client}")
        await connection_manager.connect(client_id, websocket)
        
        # Update client status to Active
        for attempt in range(retry_attempts):
            try:
                client.status = "Active"
                db.commit()
                logger.info(f"Client {client_id} connected successfully, status updated to 'Active'.")
                break
            except SQLAlchemyError as db_error:
                db.rollback()
                logger.error(f"Attempt {attempt + 1} - Failed to update 'Active' status for {client_id}: {db_error}")
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
                        logger.error(f"Attempt {attempt + 1} - Database error for client {client_id}: {db_error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2)
                        else:
                            raise
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected gracefully.")
                break
            except Exception as e:
                logger.error(f"Error handling message from client {client_id}: {e}")
                await websocket.send_text("An error occurred. Please try again later.")
                break
    
    except SQLAlchemyError as db_error:
        logger.error(f"Database error for client {client_id}: {db_error}")
        db.rollback()
        await websocket.close(code=1002, reason="Database error.")
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected unexpectedly.")
    except Exception as e:
        logger.error(f"Unexpected error for client {client_id}: {e}")
        await websocket.close(code=1000, reason="Internal server error.")
    
    finally:
        # Cleanup
        for attempt in range(retry_attempts):
            try:
                await connection_manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Failed to disconnect client {client_id}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
        
        # Update client status to Inactive
        if client:
            for attempt in range(retry_attempts):
                try:
                    db.refresh(client)
                    client.status = "Inactive"
                    db.commit()
                    logger.info(f"Client {client_id} is now inactive. DB updated successfully.")
                    break
                except SQLAlchemyError as db_error:
                    db.rollback()
                    logger.error(f"Attempt {attempt + 1} - Failed to update 'Inactive' status for {client_id}: {db_error}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2)
                    else:
                        raise
        
        logger.info(f"Cleanup completed for client {client_id}.")

# Diagnostic endpoint to check scheduler status
@app.get("/scheduler-status")
async def scheduler_status():
    if not scheduler:
        return {"status": "not_initialized", "jobs": []}
    
    if not scheduler.running:
        return {"status": "not_running", "jobs": []}
    
    try:
        jobs = scheduler.get_jobs()
        job_info = []
        for job in jobs:
            job_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time),
                "trigger": str(job.trigger),
                "pending": getattr(job, 'pending', 'unknown')
            })
        
        return {
            "status": "running",
            "job_count": len(jobs),
            "jobs": job_info,
            "scheduler_state": str(scheduler.state) if hasattr(scheduler, 'state') else 'unknown'
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"status": "error", "error": str(e)}

# Manual trigger endpoints for testing - Fixed HTTP methods
@app.post("/trigger-db-ping")
async def trigger_database_ping():
    logger.info("=== MANUAL TRIGGER: Database ping ===")
    try:
        result = await ping_database()
        return {"status": "success", "message": "Database ping executed", "result": result}
    except Exception as e:
        logger.error(f"Manual database ping failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/trigger-aggregation")
async def trigger_weight_aggregation():
    logger.info("=== MANUAL TRIGGER: Weight aggregation ===")
    try:
        result = await scheduled_aggregate_weights()
        return {"status": "success", "message": "Weight aggregation executed", "result": str(result)}
    except Exception as e:
        logger.error(f"Manual weight aggregation failed: {e}")
        return {"status": "error", "message": str(e)}

# Debug endpoint to check scheduler internals
@app.get("/debug-scheduler")
async def debug_scheduler():
    if not scheduler:
        return {"error": "Scheduler not initialized"}
    
    try:
        return {
            "scheduler_running": scheduler.running,
            "scheduler_state": str(scheduler.state) if hasattr(scheduler, 'state') else 'unknown',
            "job_stores": list(scheduler._jobstores.keys()) if hasattr(scheduler, '_jobstores') else [],
            "executors": list(scheduler._executors.keys()) if hasattr(scheduler, '_executors') else [],
            "jobs_detail": [
                {
                    "id": job.id,
                    "name": job.name,
                    "func_name": job.func.__name__,
                    "next_run": str(job.next_run_time),
                    "trigger": str(job.trigger),
                    "pending": getattr(job, 'pending', 'unknown'),
                    "max_instances": getattr(job, 'max_instances', 'unknown')
                }
                for job in scheduler.get_jobs()
            ]
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {"error": str(e)}

# Also ensure cleanup on process exit as backup
atexit.register(lambda: scheduler.shutdown(wait=False) if (scheduler and scheduler.running) else None)