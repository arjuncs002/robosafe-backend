from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./robosafe.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(Float, index=True)
    count = Column(Integer)
    source = Column(String, default="webcam")

class ControlCommand(Base):
    __tablename__ = "control_commands"
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(Float, index=True)
    command = Column(String)

class MMWaveState(Base):
    __tablename__ = "mmwave_states"
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(Float, index=True)
    status = Column(String)
    presence = Column(Integer)
    energy_delta = Column(Integer)
    respiration_detected = Column(Boolean)

def init_db():
    Base.metadata.create_all(bind=engine)