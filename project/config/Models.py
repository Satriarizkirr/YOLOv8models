from sqlalchemy import Column, Integer, String, DECIMAL, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Detection(Base):
    __tablename__ = 'detection'

    id = Column(Integer, primary_key=True, autoincrement=True)
    detected_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    class_ = Column(String(100))  
    confidence = Column(DECIMAL(5, 4))
    image_path = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_esp_32 = Column(String(55), nullable=False) 
    ip_esp_8266 = Column(String(55), nullable=False) 
    username = Column(String(100), nullable=False)
    password = Column(String(100), nullable=False)  
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())