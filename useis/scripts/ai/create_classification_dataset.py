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
output_data_dir = Path('/data_1/classification_dataset_2')
output_data_dir.mkdir(parents=True, exist_ok=True)
sampling_rate = 6000
# num_threads = int(np.ceil(cpu_count() - 10))
num_threads = 10
replication_level = 5
snr_threshold = 10
sequence_length_second = 2
perturbation_range_second = 1
image_width = 128
image_height = 128
buffer_image_fraction = 0.05

buffer_image_sample = int(image_width * buffer_image_fraction)

hop_length = int(sequence_length_second * sampling_rate //
                 (image_width + 2 * buffer_image_sample))


# def spectrogram(trace: Trace, perturbation_sample: float = None):
#
#     mel_spec = MelSpectrogram(sample_rate=sampling_rate,
#                               n_mels=image_height,
#                               hop_length=hop_length,
#                               power=1,
#                               pad_mode='reflect',
#                               normalized=True)
#
#     amplitude_to_db = AmplitudeToDB()
#
#     trace = trace.taper(max_length=0.01, max_percentage=0.05)
#     # plt.figure(3)
#     # plt.clf()
#     # plt.plot(trace.data)
#
#     n_pts = sequence_length_second * sampling_rate
#
#     target_sequence_length = int(
#         2 ** (np.ceil(np.log(n_pts) / np.log(2))))
#     n_zeros = target_sequence_length - len(trace.data)
#
#     data = np.hstack((trace.data, np.zeros(n_zeros)))
#
#     data = np.roll(data, perturbation_sample)
#
#     # data = data[:sequence_length_second * sampling_rate]
#
#     torch_data = torch.tensor(data).type(torch.float32)
#
#     spec = (mel_spec(torch_data) + 1e-3)[:, buffer_image_sample:
#                                          buffer_image_sample
#                                          + image_width]
#     spec_db = amplitude_to_db(spec.abs())
#     spec_db = spec_db - spec_db.min()
#     spec_db = (spec_db / spec_db.max() * 255).type(torch.uint8)
#     img = Image.fromarray(np.array(spec_db.tolist()).astype(
#         np.uint8))
#
#     return img


def create_training_image(waveform_file):
    # logger.info('here')
    waveform_file_path = input_data_dir / waveform_file
    cat = read_events(waveform_file_path.with_suffix('.xml'))
    st = read(waveform_file_path)

    if cat[0].preferred_origin() is not None:
        origin = cat[0].preferred_origin()
    else:
        origin = cat[0].origins[-1]

    if origin.evaluation_mode == 'automatic':
        return

    event_time = origin.time.timestamp

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
    logger.info(event_type_lookup[cat[0].event_type])
    if event_type_lookup[cat[0].event_type] == 'seismic event':
        for arrival in cat[0].preferred_origin().arrivals:
            site = arrival.pick.site
            for tr in st.select(site=site).copy():
                # filtering out low frequency before calculating the SNR
                tr = tr.filter('highpass', freq=100)
                snr = calculate_snr(tr, arrival.pick.time,
                                    pre_wl=20e-3,
                                    post_wl=20e-3)
                if arrival.pick.evaluation_mode == 'automatic':
                    if snr < snr_threshold:
                        continue
                tr.trim(starttime=tr.stats.starttime,
                        endtime=tr.stats.starttime +
                        sequence_length_second,
                        pad=True,
                        fill_value=0)
                trs.append(tr.copy().resample(sampling_rate=
                                              int(sampling_rate)))

    elif event_type_lookup[cat[0].event_type] == 'blast':
        for tr in st:
            if (np.max(np.abs(tr.data)) / np.std(tr.data)) < 6:
                continue
            trs.append(tr.trim(endtime=tr.stats.starttime +
                               sequence_length_second))

    else:
        for tr in st:
            trs.append(tr.trim(endtime=tr.stats.starttime +
                               sequence_length_second))

    ct = 0
    for tr in trs:
        try:
            spec = spectrogram(tr)
        except Exception as e:
            logger.error(e)
            ct += 1
            logger.info(f'{ct} exceptions')
            return
        spec = (spec / np.max(spec)) * 255
        img = Image.fromarray(np.array(spec.tolist()).astype(np.uint8))

        filename = f'{event_time:0.0f}_{tr.stats.site}_' \
                   f'{tr.stats.channel}.jpg'

        out_dir = (output_data_dir /
                   event_type_lookup[cat[0].event_type])

        out_dir.mkdir(parents=True, exist_ok=True)

        out_file_name = out_dir / filename
        ImageOps.grayscale(img).save(out_file_name, format='JPEG')


if __name__ == '__main__':
    file_list = [f for f in input_data_dir.glob('*.mseed')]
    for f in tqdm(file_list):
        create_training_image(f)

    # file_list = [f for f in input_data_dir.glob('*.mseed')]
    # with Pool(4) as pool:
    #     event_list = list(tqdm(pool.map(create_training_image, file_list),
    #                            total=len(file_list)))
    #     # with tqdm(total=len(file_list)) as p_bar:
    #     #     for i, _ in enumerate(pool.imap(create_training_image, file_list)):
    #     #         logger.info('here')
    #     #         p_bar.update()
