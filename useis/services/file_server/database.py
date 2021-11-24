from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime
from uquake.core.stream import Trace

Base = declarative_base()


class Database(object):
    def __init__(self, url, port, username, password, db):
        self.connection_string=f'postgresql://{username}:{password}' \
                               f'@{url}:{port}/db'
        self.engine = create_engine(self.connection_string)
        Base.metadata.create_all(self.engine)
        self.db_session = sessionmaker(bind=self.engine)
        self.session = self.db_session()

    def index_trace(self, tr: Trace, file_path):
        csd = ContinuousSeismicData(network=tr.stats.network,
                                    station=tr.stats.station,
                                    location=tr.stats.location,
                                    channel=tr.stats.channel,
                                    start_time=tr.stats.starttime.datetime,
                                    end_time=tr.stats.endtime.datetime,
                                    filepath=file_path)
        self.session.add(csd)
        self.session.commit()


class ContinuousSeismicData(Base):
    __tablename__ = "continuous_seismic_data"
    id = Column(Integer, primary_key=True, index=True)
    network = Column(String, index=True)
    station = Column(String, index=True)
    location = Column(String, index=True)
    channel = Column(String, index=True)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime, index=True)
    filepath = Column(String)


