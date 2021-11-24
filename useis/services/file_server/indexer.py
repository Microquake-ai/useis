from os import environ, walk
from glob import glob
from uquake.core import read
from .database import Database

data_path = environ['UQ_FS_DATA_PATH']
database = environ['UQ_FS_DATABASE']
db_user = environ['UQ_FS_DB_USER']
db_password = environ['UQ_FS_DB_PASSWORD']
db_url = environ['UQ_FS_DB_URL']
db_port = environ['UQ_FS_DB_PORT']

print(len(glob(data_path + '/*', recursive=True)))

db = Database(db_url, db_port, db_user, db_password, database)

for (dirpath, dirnames, filenames) in walk(data_path):
    for filename in filenames:
        st = read(dirpath + '/' + filename)

        for tr in st:

# for file in glob(data_path, recursive=True):



