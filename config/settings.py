import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env.server')

class Settings:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Azure Storage
    CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
    SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
    CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
    SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
    LOCAL_CONTAINER_URL = os.getenv("LOCAL_CONTAINER_URL")
    GLOBAL_CONTAINER_URL = os.getenv("GLOBAL_CONTAINER_URL")
    CLIENT_CONTAINER_SAS_TOKEN = os.getenv("CLIENT_CONTAINER_SAS_TOKEN")
    CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")
    
    # Security
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "your_admin_secret_key")
    
    # Constants
    ARCH_BLOB_NAME = "model_architecture.h5"
    
    @classmethod
    def validate(cls):
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is missing")
        if not cls.CLIENT_ACCOUNT_URL or not cls.SERVER_ACCOUNT_URL:
            raise ValueError("SAS url environment variable is missing.")

settings = Settings()
settings.validate()