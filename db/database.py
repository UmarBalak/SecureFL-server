from sqlalchemy import create_engine, Column, String, DateTime, Table, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
from threading import Lock
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

engine = None
engine_lock = Lock()

def initialize_engine():
    global engine
    with engine_lock:
        if engine is not None:
            engine.dispose()
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=90,  # Recycle every 90 seconds
            pool_size=3,
            max_overflow=5,
            pool_timeout=20
        )

initialize_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

client_model_association = Table(
    'client_model_association', Base.metadata,
    Column('client_id', String, ForeignKey('clients.client_id')),
    Column('model_id', Integer, ForeignKey('global_models.id'))
)

class Client(Base):
    __tablename__ = "clients"
    csn = Column(String, primary_key=True)
    client_id = Column(String, unique=True, nullable=False)
    api_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Inactive")
    contribution_count = Column(Integer, default=0)
    models_contributed = relationship("GlobalModel", secondary=client_model_association, back_populates="clients")

class GlobalModel(Base):
    __tablename__ = "global_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_clients_contributed = Column(Integer, default=0)
    client_ids = Column(String)
    clients = relationship("Client", secondary=client_model_association, back_populates="models_contributed")

class GlobalAggregation(Base):
    __tablename__ = "global_aggregation"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except OperationalError as e:
        print(f"Database connection error: {e}")
        initialize_engine()
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()