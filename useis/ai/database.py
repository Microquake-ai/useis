from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Define a declarative base for the ORM
Base = declarative_base()


# Define a class to represent the record table
class Record(Base):
    __tablename__ = 'records'
    id = Column(Integer, primary_key=True)
    image10s_filename = Column(String)
    image3s_filename = Column(String)
    image1s_filename = Column(String)
    Synthetic = Column(Boolean)
    magnitude = Column(Float)
    sampling_rate = Column(Float)
    categories = Column(String)
    event_id = Column(Integer)
    sensor_type = Column(String)
    bounding_box_start = Column(Integer)
    bounding_box_end = Column(Integer)


def connect(db_url):
    # Create an engine to connect to the database
    engine = create_engine('sqlite:///mydatabase.db', echo=True)
    # Create the table in the database
    Base.metadata.create_all(engine)

    # Create a session to interact with the database
    Session = sessionmaker(bind=engine)
    session = Session()

    return session


# # Define the data to insert into the database
# data = [
#     Event(
#         image10s_filename='image10s.jpg',
#         image3s_filename='image3s.jpg',
#         image1s_filename='image1s.jpg',
#         categories='category1',
#         event_id=1,
#         bounding_box=[0, 0, 100, 100]
#     ),
#     Event(
#         image10s_filename='image10s.jpg',
#         image3s_filename='image3s.jpg',
#         image1s_filename='image1s.jpg',
#         categories='category2',
#         event_id=2,
#         bounding_box=[50, 50, 150, 150]
#     ),
#     Event(
#         image10s_filename='image10s.jpg',
#         image3s_filename='image3s.jpg',
#         image1s_filename='image1s.jpg',
#         categories='category1',
#         event_id=3,
#         bounding_box=[100, 100, 200, 200]
#     )
# ]


class DBManager(object):
    def __init__(self, db_url):
        self.db_url = db_url
        session = connect(db_url)
        session.close()

    def add_record(self, image10s_filename, image3s_filename, image1s_filename,
                   sampling_rate, categories, event_id, bounding_box):
        record = Record(image10s_filename, image3s_filename, image1s_filename,
                        sampling_rate, categories, event_id, bounding_box)
        session = connect(self.db_url)

        session.add_all([record])
        session.commit()
        session.close()


