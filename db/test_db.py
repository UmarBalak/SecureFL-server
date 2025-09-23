from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
import logging
from datetime import datetime
import time

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='.env.server')
DATABASE_URL = os.getenv("DATABASE_URL")
    

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create engine with connection pooling and pre-ping
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_db_connection():
    logger.info("Testing database connection...")
    try:
        # Test raw connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info(f"Raw connection test successful: {result.scalar()}")
        
        # Test ORM session
        with SessionLocal() as db:
            # Simple query to verify table access
            result = db.execute(text("SELECT version FROM global_models ORDER BY version DESC LIMIT 1"))
            version = result.scalar()
            logger.info(f"ORM query successful. Latest version: {version if version is not None else 'No models found'}")
        
        # Simulate prolonged connection
        logger.info("Testing connection stability over 30 seconds...")
        with SessionLocal() as db:
            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < 30:
                db.execute(text("SELECT 1"))
                logger.info("Connection alive")
                time.sleep(5)
    
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        logger.info("Closing connections")
        engine.dispose()

if __name__ == "__main__":
    test_db_connection()