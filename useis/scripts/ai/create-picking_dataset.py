import pandas
from uquake.core import read, read_events, read_inventory, UTCDateTime
from uquake.core.trace import Trace
from uquake.core.logging import logger
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import pickle
from useis.core.project_manager import ProjectManager
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import torch
from uquake.waveform.pick import calculate_snr
from PIL import Image, ImageOps
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from dataset import spectrogram


def get_waveform(waveform_file, inventory):
    st = read(waveform_file)
    for i, tr in enumerate(st):
        for sensor in inventory.sensors:
            if sensor.alternate_code == tr.stats.station:
                st[i].stats.network = inventory[0].code
                st[i].stats.station = sensor.station.code
                st[i].stats.location = sensor.location_code
                for channel in sensor.channels:
                    if tr.stats.channel in channel.code:
                        st[i].stats.channel = channel.code
                        break
                break
    return st


def get_event_id(filename):
    cat = read_events(filename)
    return str(filename.stem), cat[0].resource_id.id


event_type_lookup = {'anthropogenic event': 'noise',
                     'acoustic noise': 'noise',
                     'reservoir loading': 'noise',
                     'road cut': 'noise',
                     'controlled explosion': 'blast',
                     'quarry blast': 'blast',
                     'earthquake': 'seismic event',
                     'sonic boom': 'noise',
                     'collapse': 'noise',
                     'other event': 'noise',
                     'thunder': 'noise',
                     'induced or triggered event': 'noise',
                     'explosion': 'blast',
                     'experimental explosion': 'blast'}

input_data_dir = Path('/data_1/ot-reprocessed-data/')
output_data_dir = Path('/data_1/pick_dataset')
output_data_dir.mkdir(parents=True, exist_ok=True)
sampling_rate = 6000
# num_threads = int(np.ceil(cpu_count() - 10))
num_threads = 10
replication_level = 5
snr_threshold = 10
sequence_length_second = 2
perturbation_range_second = 1
number_sample = 256
buffer_image_fraction = 0.05


def create_training_signal(waveform_file):
    # logger.info('here')
    waveform_file_path = input_data_dir / waveform_file
    cat = read_events(waveform_file_path.with_suffix('.xml'))
    st = read(waveform_file_path)

    event_time = cat[0].preferred_origin().time.timestamp

    try:
        st = st.detrend('demean').detrend(
            'linear').taper(max_percentage=0.1,
                            max_length=0.01).resample(
            sampling_rate=sampling_rate)
    except Exception as e:
        logger.error(e)
        trs = []
        for tr in st:
            if np.nan in tr.data:
                continue
            trs.append(tr.copy())
        st.traces = trs

    trs = []
    snrs = []
    pick_times = []
    logger.info(event_type_lookup[cat[0].event_type])
    if event_type_lookup[cat[0].event_type] == 'seismic event':
        for arrival in cat[0].preferred_origin().arrivals:
            site = arrival.pick.site
            for tr in st.select(site=site):
                snr = calculate_snr(tr, arrival.pick.time,
                                    pre_wl=20e-3,
                                    post_wl=20e-3)
                if snr < snr_threshold:
                    continue
                tr.trim(starttime=tr.stats.starttime,
                        endtime=tr.stats.starttime +
                        sequence_length_second,
                        pad=True,
                        fill_value=0)
                trs.append(tr.copy().resample(sampling_rate=
                                              int(sampling_rate)))
                snrs.append(snr)
                pick_times.append(arrival.pick.time)

    else:
        return

    ct = 0
    for tr, pick_time in zip(trs, pick_times):
        for i in range(0, 5):
            try:

                start = np.random.randint(0, number_sample -
                                          int(0.25 * number_sample))

                pick_sample = int((pick_time - tr.stats.starttime) *
                                  tr.stats.sampling_rate)

                start_sample = int(pick_sample - start)

                data = tr.data[start_sample: start_sample + number_sample]
                data = data - np.mean(data)
                data = np.abs(data) / np.abs(data).max()
                pick = start

                out_dict = {'data': data, 'pick': pick}

                out_dir = output_data_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f'{event_time:0.0f}_{tr.stats.site}_' \
                           f'{tr.stats.channel}_{i}.pickle'

                with open(out_dir / filename, 'wb') as file_out:
                    pickle.dump(out_dict, file_out)
            except Exception as e:
                logger.error(e)


if __name__ == '__main__':
    file_list = [f for f in input_data_dir.glob('*.mseed')]
    for f in tqdm(file_list):
        create_training_signal(f)

    # file_list = [f for f in input_data_dir.glob('*.mseed')]
    # with Pool(4) as pool:
    #     event_list = list(tqdm(pool.map(create_training_image, file_list),
    #                            total=len(file_list)))
    #     # with tqdm(total=len(file_list)) as p_bar:
    #     #     for i, _ in enumerate(pool.imap(create_training_image, file_list)):
    #     #         logger.info('here')
    #     #         p_bar.update()
