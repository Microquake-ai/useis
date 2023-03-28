from torch.utils.data import Dataset
from pathlib import Path
from uquake import read, read_events
from uquake.core import UTCDateTime
from uquake.core.logging import logger
from uquake.core.stream import Stream
from uquake.core.event import Event, Catalog
import numpy as np
from obspy.signal.trigger import (recursive_sta_lta, plot_trigger, coincidence_trigger,
                                  trigger_onset)
import matplotlib.pyplot as plt
from useis.ai.model import generate_spectrogram
from PIL import Image, ImageOps
from useis.ai.database import Record
from event_type_lookup import event_type_lookup
from scipy.stats import skew, kurtosis

from useis.processors import classifier
from importlib import reload
reload(classifier)
# from useis.processors.classifier import Classifier
from ipdb import set_trace
import multiprocessing
from tqdm import tqdm

np.random.seed(124)

##
# Frame size

raw_data_path = Path('/data_1/ot-reprocessed-data')
classifier_project = classifier.Classifier('/data_1/projects/', 'classifier', 'OT',
                                           reset_training=False)
training_data_path = classifier_project.paths.training_dataset
# training_data_path.mkdir(parents=True, exist_ok=True)

context_trace = raw_data_path.glob('*.context_mseed')

window_length_seconds = np.floor(np.logspace(0, 1, 3))
sampling_rates = np.array([6000])

frame_size = 1  # frame size in second HNAS request data every second from the Paladin


def test_impulsiveness(trace):

    # Load the signal from a file or generate it using numpy
    signal = trace.detrend('demean').detrend('linear').data
    signal = signal / np.max(np.abs(signal))

    # Compute the signal statistics
    std = np.std(signal)
    mad = np.median(np.abs(signal - np.median(signal)))
    # skewness = skew(signal)
    # kurt = kurtosis(signal)
    # print(mad, skewness, kurt)

    # Classify the signal based on its statistics
    if mad/std > 1e-3:
        return False
    else:
        return True


def window_signal(stream: Stream, trigger_on: float = 0.5, trigger_off: float = 0.1,
                  sta: float = 0.5, lta: float = 2):
    stream = stream.composite()
    for i, tr in enumerate(stream):
        stream[i].data /= np.std(stream[0].data)
    df = stream[0].stats.sampling_rate
    data = np.hstack([stream[0].data[-1::-1], stream[0].data])
    cft = recursive_sta_lta(data, int(sta * df), int(lta * df))
    cft = cft[len(cft)-len(stream[0].data):]
    triggers = trigger_onset(cft / np.nanmax(cft), trigger_on, trigger_off)

    start_time = stream[0].stats.starttime + triggers[0][0] / df
    end_time = stream[0].stats.starttime + triggers[-1][1] / df

    return start_time, end_time


# def create_spectrograms(stream: Stream, event_type: str, end_time: UTCDateTime,
#                         magnitude: float, expect_impulsive=True):
#
#     init_sampling_rate = stream[0].stats.sampling_rate
#     # from ipdb import set_trace
#     # set_trace()
#     starttime = stream[0].stats.starttime
#     end_time_sample = (end_time - starttime) * init_sampling_rate
#     for sampling_rate in sampling_rates:
#         end_time_resampled = starttime + end_time_sample / sampling_rate
#         for window_length in window_length_seconds:
#             trs_resampled = []
#             for tr in stream.copy():
#                 # if expect_impulsive:
#                 #     if not test_impulsiveness(tr):
#                 #         continue
#                 tr_resampled = tr.copy()
#                 tr_resampled.stats.sampling_rate = sampling_rate
#                 trs_resampled.append(tr_resampled)
#
#             if not trs_resampled:
#                 continue
#
#             st_resampled = Stream(traces=trs_resampled)
#             # st_resampled = stream.copy().resample(sampling_rate)
#             start_time = end_time_resampled - window_length
#             st_trimmed = st_resampled.copy().trim(starttime=start_time,
#                                                   endtime=end_time_resampled,
#                                                   pad=True, fill_value=0)
#             st_trimmed = st_trimmed.taper(max_percentage=0.01, max_length=1e-3)
#
#             # from ipdb import set_trace
#             # set_trace()
#
#             spectrograms = generate_spectrogram(st_trimmed)
#
#             filenames = write_spectrogram(spectrograms, event_type, end_time_resampled,
#                                           magnitude, window_length)
#
#             for filename, tr in zip(filenames, st_trimmed):
#                 plt.clf()
#                 plt.plot(tr.data, 'k')
#                 plt.axis('off')
#                 plt.tight_layout()
#                 filename = Path(filename)
#                 classifier_dataset_path = classifier_project.paths.training_dataset
#                 plt.savefig(f'{classifier_dataset_path / event_type / filename.stem}'
#                             f'_td.png')


# def write_spectrogram(spectrograms, event_type, end_time, magnitude, duration):
#
#     filenames = []
#     for i, spectrogram in enumerate(spectrograms):
#         if np.mean(spectrogram) == 0:
#             continue
#
#         filename = f'{event_type}_{end_time}_mag{magnitude:0.2f}_{duration}sec_ch{i}.png'
#         data_path = training_data_path / event_type
#         data_path.mkdir(parents=True, exist_ok=True)
#         path = training_data_path / event_type / filename
#         spectrogram.save(path, format='png')
#         filenames.append(filename)
#
#     return filenames


def create_blast_training_set(stream: Stream, event: Event, filename: str):
    # find the start and end of the blast train using sta/lta

    event_id = filename.name

    blast_start_time, blast_end_time = window_signal(stream)

    end_time = blast_start_time + 0.5
    ct = 1

    original_event_type = event.event_type

    while end_time <= blast_end_time:
        classifier_project.create_spectrogram_training(stream, 'blast',
                                                       original_event_type,
                                                       event_id, end_time,
                                                       event.magnitudes[-1].mag)
        end_time += 1


def find_end_signal(stream: Stream, event: Event):
    # finding the start and end of the signal
    p_pick_time = None
    for arrival in event.preferred_origin().arrivals:
        pick = arrival.get_pick()
        if pick.site == stream[0].stats.site:
            if arrival.phase.upper() == 'P':
                p_pick_time = pick.time

    if p_pick_time is None:
        tt_grid = classifier_project.travel_times.select(
            seed_labels=[stream[0].stats.site], phase='P')[0]
        p_pick_time = event.preferred_origin().time + \
                      tt_grid.interpolate(event.preferred_origin().loc)[0]

    # estimating the event duration
    # measuring the background energy

    # background = stream.copy().trim(starttime=p_pick_time-0.5, endtime=p_pick_time-0.1)
    # background_energy = np.mean(background[0].data ** 2)
    #
    # signal = stream.copy().trim(starttime=p_pick_time).detrend('demean').detrend(
    #     'linear')[0].data
    #
    # energy_window = 100  # sample
    # for i in range(len(signal) - energy_window):
    #     energy_remaining = np.mean(signal[i:i+energy_window] ** 2)
    #     if energy_remaining < 10 * background_energy:
    #         end_signal_t = p_pick_time + i / stream[0].stats.sampling_rate
    #         break

    start_signal = p_pick_time
    end_signal = None

    return start_signal, end_signal


def create_seismic_event_training_set(stream: Stream, event: Event, filename: str):

    start_signal, end_signal = find_end_signal(stream, event)
    # duration = end_signal - start_signal
    # min_start_window = start_signal + 0.3 * duration - np.min(window_length_seconds)
    # max_end_window = end_signal - 0.3 * duration + np.min(window_length_seconds)

    event_type = event_type_lookup[event.event_type]

    window_range = 0.6
    window_offset = 0.2

    perturbations = np.random.rand(5) * window_range + window_offset
    end_signals = [start_signal + t for t in perturbations]
    event_id = filename.name

    original_event_type = event.event_type

    for end_signal in end_signals:
        if event.preferred_magnitude():
            magnitude = event.preferred_magnitude().mag
        else:
            magnitude = event.magnitudes[-1].mag
        classifier_project.create_spectrogram_training(stream.copy(), event_type,
                                                       original_event_type,
                                                       event_id, end_signal,
                                                       magnitude,
                                                       simulate_magnitude=True)


def create_noise_training_set(stream: Stream, event: Event, filename: Path):
    end_signal = stream[0].stats.starttime + 12
    event_id = filename.name
    original_event_type = event.event_type
    classifier_project.create_spectrogram_training(stream, 'noise', original_event_type,
                                                   event_id, end_signal,
                                                   event.magnitudes[-1].mag,
                                                   expect_impulsive=False)


def event_size_scaling(stream: Stream, event: Event, end_signal, output_number=3,
                       magnitude_range=[-1, 3], magnitude_resolution=0.01, bvalue=1,
                       scaling_factor=2):
    """
    The function transform the stream to simulate larger magnitude.
    The scaling is performed by altering the frequency content of the waveform.
    The corner frequency is reduced by a factor 3 for every unit of magnitude.
    Some reference on why a factor of 3 is used can be found in
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JB003084

    other authors have suggested a scaling to be closer to 2.

    Note that the distribution of magnitude produced in uniform. This is to counter-
    balance the typical distribution that is skewed towards smaller magnitude.

    @param stream: the waveform
    @type stream: uquake.core.stream.Stream
    @param event: the event information
    @type event: uquake.core.event.Event
    @param output_number: number of output.
    @param magnitude_range: the range of magnitude to randomly select from
    @param magnitude_resolution: the magnitude resolution
    @param bvalue: The b-value of the natural distribution to counterbalance
    @param scaling_factor: determine how the frequency scales with the increase or
    decrease in magnitude.
    """
    try:
       magnitude = event.preferred_magnitude().mag
    except Exception as e:
        logger.warning(e)
        magnitude = event.magnitudes[-1].mag
    if magnitude == -999:
        return
    streams = []
    mags = []
    end_signals = []
    for i in range(output_number):
        target_mag = generate_random_magnitude(range=magnitude_range,
                                               resolution=magnitude_resolution,
                                               bvalue=bvalue)
        mags.append(target_mag)
        frequency_scaling = scaling_factor ** (target_mag - magnitude)
        trs = []
        for tr in stream.copy():
            tmp = (end_signal - tr.stats.starttime) * tr.stats.sampling_rate
            end_signal = tr.stats.starttime + tmp / frequency_scaling
            tr.stats.sampling_rate /= frequency_scaling
            trs.append(tr)
            end_signals.append(end_signal)
        streams.append(Stream(traces=trs))
    return {'streams': streams, 'magnitudes': mags, 'end_signals': end_signals}


def generate_random_magnitude(range=[-2, 2.3]):
    r = np.max(range) - np.min(range)
    return np.random.rand(1)[0] * r + np.min(range)


durations = []
magnitudes = []


def create_training_dataset(context_trace):
    try:

        event_id = context_trace.name
        if classifier_project.event_exists(event_id):
            logger.info('event already exists in the database')
            return

        seismogram_file = context_trace

        try:
            st = read(context_trace).detrend('demean').detrend('linear')
        except Exception as e:
            logger.error(e)
            return
        event = read_events(context_trace.with_suffix('.xml'))[0]
        if np.sum(st.composite()[0].data) / len(st.composite()[0].data) > 0.5:
            return

        if event_type_lookup[event.event_type] == 'blast':
            create_blast_training_set(st, event, seismogram_file)

        elif event_type_lookup[event.event_type] == 'seismic event':

            create_seismic_event_training_set(st.copy(), event, seismogram_file)

        elif 'noise' in event_type_lookup[event.event_type]:
            create_noise_training_set(st, event, seismogram_file)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    mseed_files = [f for f in raw_data_path.glob('*.context_mseed')]

    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(10)

    # list(tqdm(pool.imap(create_training_dataset, mseed_files),
    #           total=len(mseed_files)))

    for context_trace in tqdm(mseed_files):
        # try:
        create_training_dataset(context_trace)
        # except Exception as e:
        #     logger.error(e)
        # # print(context_trace)


