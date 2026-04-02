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


# ─── PHASE 1 MODELS ───────────────────────────────────────────────────────────

class Detection(Base):
    __tablename__ = "detections"
    id     = Column(Integer, primary_key=True, index=True)
    ts     = Column(Float,   index=True)
    count  = Column(Integer)
    source = Column(String,  default="webcam")


class ControlCommand(Base):
    __tablename__ = "control_commands"
    id      = Column(Integer, primary_key=True, index=True)
    ts      = Column(Float,   index=True)
    command = Column(String)


class MMWaveState(Base):
    __tablename__ = "mmwave_states"
    id                   = Column(Integer, primary_key=True, index=True)
    ts                   = Column(Float,   index=True)
    status               = Column(String)
    presence             = Column(Integer)
    energy_delta         = Column(Integer)
    respiration_detected = Column(Boolean)


# ─── PHASE 2 MODELS ───────────────────────────────────────────────────────────

class MapFlag(Base):
    __tablename__ = "map_flags"
    id        = Column(Integer, primary_key=True, index=True)
    ts        = Column(Float,   index=True)
    flag_type = Column(String)   # detected | arrived | alive | not_alive
    x         = Column(Float,   default=0.0)
    y         = Column(Float,   default=0.0)
    label     = Column(String,  default="")


class RoverPosition(Base):
    __tablename__ = "rover_positions"
    id      = Column(Integer, primary_key=True, index=True)
    ts      = Column(Float,   index=True)
    x       = Column(Float,   default=0.0)
    y       = Column(Float,   default=0.0)
    heading = Column(Float,   default=0.0)


class DriveMode(Base):
    __tablename__ = "drive_mode"
    id   = Column(Integer, primary_key=True, index=True)
    mode = Column(String,  default="MANUAL")


class AutoCommand(Base):
    __tablename__ = "auto_commands"
    id      = Column(Integer, primary_key=True, index=True)
    ts      = Column(Float,   index=True)
    command = Column(String,  default="STOP")


class LifeConfirm(Base):
    """
    One-row table. Jetson polls this.
    confirmed=True  → operator has responded.
    result          → 'alive' | 'not_alive'
    cleared         → Jetson has acknowledged and reset.
    """
    __tablename__ = "life_confirm"
    id        = Column(Integer, primary_key=True, index=True)
    confirmed = Column(Boolean, default=False)
    result    = Column(String,  default="")
    cleared   = Column(Boolean, default=True)


def init_db():
    Base.metadata.create_all(bind=engine)