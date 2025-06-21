from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

 

def create_session():
    engine = create_engine('mysql://root:@localhost/yolo_v8')
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
