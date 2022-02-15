from ..core.project_manager import ProjectManager
import pandas as pd

from pathlib import Path
import os

from uquake.core import read, UTCDateTime
from multiprocessing import Pool, cpu_count
from uquake.core.logging import logger
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from uquake.core.stream import Stream


def extract_trace_info_mseed(mseed_file):
    try:
        st = read(mseed_file)
    except Exception as e:
        logger.warning(e)
        return []
    output_list = []
    for tr in st:

        output_dict = {'mseed_file': mseed_file,
                       'start_time': tr.stats.starttime.datetime,
                       'start_time_timestamp': tr.stats.starttime.timestamp,
                       'end_time': tr.stats.endtime.datetime,
                       'end_time_timestamp': tr.stats.endtime.timestamp,
                       'network': tr.stats.network,
                       'station': tr.stats.station,
                       'location': tr.stats.location,
                       'channel': tr.stats.channel}

        output_list.append(output_dict)

    return output_list


def change_channel_name(mseed_file):
    from uquake.core.stream import Stream
    try:
        st = read(mseed_file)
    except Exception as e:
        logger.warning(e)
        return []
    trs = []
    for tr in st.copy():
        if 'N' == tr.stats.channel[-1]:
            tr.stats.channel = tr.stats.channel.replace('N', '1')
        elif 'E' == tr.stats.channel[-1]:
            tr.stats.channel = tr.stats.channel.replace('E', '2')
        else:
            tr.stats.channel = tr.stats.channel.replace('Z', '3')

        trs.append(tr)

    output_mseed = Path(str(mseed_file).replace('data_sept',
                                                'data_sept_reprocessed'))

    output_mseed.parent.mkdir(exist_ok=True, parents=True)

    Stream(traces=trs).write(output_mseed)
    return mseed_file


class ContinuousWaveformDataExtractor(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, data_path: Path,
                 create_waveform_file_index: bool = False,
                 multiprocessing_indexing: bool = True,
                 cpu_utilisation: float = 0.9):

        super().__init__(base_projects_path, project_name, network_code)
        self.paths.waveform_data = data_path

        self.files.waveform_file_index = self.paths.index / \
                                'continuous_waveform_file.index'

        self.waveform_file_index = None
        self.__cpu_count__ = cpu_count()
        self.__cpu_utilisation__ = cpu_utilisation
        if self.files.waveform_file_index.exists():
            logger.info(f'reading file index '
                        f'{self.files.waveform_file_index}')
            self.waveform_file_index = pd.read_csv(
                self.files.waveform_file_index,
                parse_dates=['start_time', 'end_time'])

        if self.waveform_file_index is None:
            if create_waveform_file_index:
                self.create_index()
            else:
                logger.warning('waveform file index does not exist and will '
                                'not be created')

    def create_index(self, multiprocessing: bool = True):

        num_threads = int(np.ceil(self.__cpu_utilisation__ * cpu_count()))
        #
        # file_index = {'starttime': [],
        #               'endtime': [],
        #               'network': [],
        #               'station': [],
        #               'location': [],
        #               'channel': []}

        wf_files = []
        for root, path, waveform_files in os.walk(self.paths.waveform_data):
            root = Path(root)
            for waveform_file in waveform_files:
                wf_files.append(root / waveform_file)

        if multiprocessing:
            with Pool(num_threads) as p:
                r = list(tqdm(p.imap(extract_trace_info_mseed,
                                      wf_files), total=len(wf_files)))
            rtmp = list(itertools.chain(*r))
            results = {key: [dic[key] for dic in rtmp]
                       for key in rtmp[0].keys()}

        logger.info('writing the file index')
        self.waveform_file_index = pd.DataFrame(results)
        self.waveform_file_index['start_time'] = \
            pd.to_datetime(self.waveform_file_index['start_time'])

        self.waveform_file_index['end_time'] = \
            pd.to_datetime(self.waveform_file_index['end_time'])

        self.waveform_file_index.to_csv(self.files.waveform_file_index)

    def get_waveforms(self, start_time, duration, network=None, station=None,
                     location=None, channel=None) -> Stream:

        start_time = UTCDateTime(start_time)

        end_time = start_time + duration
        it = self.waveform_file_index[
            (self.waveform_file_index['start_time_timestamp']
             <= end_time.timestamp)]
        it2 = it[it['end_time_timestamp'] > start_time.timestamp]

        mseed_files = np.unique(it2['mseed_file'])

        if len(mseed_files) == 0:
            raise Exception('the request did not match any file in the index')

        trs = []
        for mseed_file in mseed_files:
            st = read(mseed_file)
            for tr in st:
                trs.append(tr)

        st_out = Stream(traces=trs).merge()

        st2 = st_out.select(network=network, station=station,
                            location=location, channel=channel)
        return st2.trim(starttime=start_time, endtime=end_time)





