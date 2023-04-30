from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, MetaData,
                        distinct, Table, select)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import sqlite3
from ipdb import set_trace
import pandas as pd
import sqlite3

# Define a declarative base for the ORM
Base = declarative_base()


# Define a class to represent the record table
class Record(Base):
    __tablename__ = 'records'
    id = Column(Integer, primary_key=True)
    event_id = Column(String)
    spectrogram_filename = Column(String)
    station = Column(String)
    location = Column(String)
    channel = Column(String)
    channel_id = Column(Integer)
    magnitude = Column(Float)
    duration = Column(Float)
    end_time = Column(String)
    sampling_rate = Column(Float)
    categories = Column(String)
    original_event_type = Column(String)
    mseed_file = Column(String)
    sensor_type = Column(String)
    bounding_box_start = Column(Integer)
    bounding_box_end = Column(Integer)
    synthetic = Column(Boolean)

    def __repr__(self):
        str = ''
        for key in self.__dict__.keys():
            str += f'{key}: {self.__dict__[key]}\n'

        return str


class DBManager(object):
    def __init__(self, db_url):
        self.db_url = db_url
        self.db_path = db_url[10:]

        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.metadata = MetaData()
        Base.metadata.create_all(self.engine)
        self.table_name = Record.__tablename__

    def add_record(self, event_id=None, spectrogram_filename=None,
                   station=None, location=None, channel=None, channel_id=None,
                   magnitude=None, duration=None, end_time=None,
                   sampling_rate=None, categories=None, original_event_type=None,
                   mseed_file=None, sensor_type=None, bounding_box=None,
                   synthetic=False):

        if bounding_box is not None:
            bounding_box_start = bounding_box[0]
            bounding_box_end = bounding_box[1]
        else:
            bounding_box_start = 0
            bounding_box_end = 0

        record = Record(event_id=event_id,
                        spectrogram_filename=spectrogram_filename,
                        station=station,
                        location=location,
                        channel=channel,
                        channel_id=channel_id,
                        magnitude=magnitude,
                        duration=duration,
                        end_time=end_time,
                        sampling_rate=sampling_rate,
                        categories=categories,
                        original_event_type=original_event_type,
                        mseed_file=mseed_file,
                        sensor_type=sensor_type,
                        bounding_box_start=bounding_box_start,
                        bounding_box_end=bounding_box_end,
                        synthetic=synthetic)
        session = self.Session()

        session.add_all([record])
        session.commit()
        session.close()

    # def clear_database(self):
    #     self.metadata.reflect()
    #
    #     for table in reversed(metadata.sorted_tables):
    #         conn = engine.connect()
    #         conn.execute(table.delete())
    #         conn.close()

    def clear_database(self):
        # Get the table object based on the table name stored in the class

        table = Table(self.table_name, self.metadata, sqlite3_autoload=True,
                      autoload_with=self.engine)

        # Delete all records from the table
        delete_statement = table.delete()
        self.session.execute(delete_statement)
        self.session.commit()

    def event_exists(self, event_id):
        session = self.Session()
        query = session.query(Record).filter_by(event_id=event_id)
        record_exists = session.query(query.exists()).scalar()
        session.close()
        return record_exists

    def filter(self, **kwargs):
        session = self.Session()
        query = session.query(Record).filter_by(**kwargs)
        session.close()
        return query

    def all_unique(self, field):
        records = Record.__table__
        query = select(getattr(records.c, field))
        session = self.Session()
        results = session.execute(query).fetchall()
        unique_records = list(set(result[0] for result in results))
        session.close()
        return unique_records

    def to_pandas(self):
        con = sqlite3.connect(self.db_path)
        query = f'SELECT * FROM {self.table_name}'
        df = pd.read_sql_query(query, con)
        con.close()
        return df
        # return pd.read_sql_table(Record.__tablename__, self.engine)

    @property
    def categories(self):
        session = self.Session()
        categories = [category[0] for category in
                      session.query(distinct(Record.categories)).all()]
        session.close()
        return categories




