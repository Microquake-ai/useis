from pathlib import Path
from uquake import read, read_events
from uquake.core import UTCDateTime
from uquake.core.logging import logger
from uquake.core.stream import Stream
from uquake.core.event import Event
import numpy as np
from obspy.signal.trigger import (recursive_sta_lta, plot_trigger, coincidence_trigger,
                                  trigger_onset)
import matplotlib.pyplot as plt
from useis.ai.model import generate_spectrogram
from PIL import Image, ImageOps

from useis.processors.classifier import Classifier
classifier_project = Classifier('/data_1/projects/', 'classifier', 'OT')
##
# Frame size

raw_data_path = Path('/data_1/ot-reprocessed-data')
training_data_path = classifier_project.paths.training_dataset
training_data_path.mkdir(parents=True, exist_ok=True)

context_trace = raw_data_path.glob('*.context_mseed')

window_length_seconds = np.floor(np.logspace(0, 1, 3))
sampling_rates = np.array([1000, 2000, 6000, 10000])

frame_size = 1  # frame size in second HNAS request data every second from the Paladin

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


def create_spectrograms(stream: Stream, event_type: str, end_time: UTCDateTime):
    for sampling_rate in sampling_rates:
        for window_length in window_length_seconds:
            st_resampled = stream.copy().resample(sampling_rate)
            start_time = end_time - window_length
            st_trimmed = st_resampled.copy().trim(starttime=start_time, endtime=end_time,
                                                  pad=True, fill_value=0)

            spectrograms = generate_spectrogram(st_trimmed)

            filenames = write_spectrogram(spectrograms, event_type, window_length,
                                          sampling_rate, end_time)

            for filename, tr in zip(filenames, st_trimmed):
                plt.clf()
                plt.plot(tr.data, 'k')
                plt.axis('off')
                plt.tight_layout()
                filename = Path(filename)
                classifier_dataset_path = classifier_project.paths.training_dataset
                plt.savefig(f'{classifier_dataset_path / filename.stem}_td.png')


def write_spectrogram(spectrograms, event_type, duration, sampling_rate, end_time):

    filenames = []
    for i, spectrogram in enumerate(spectrograms):
        filename = f'{event_type}_{end_time}_ch{i}_sr{sampling_rate}_{duration}sec.png'
        path = training_data_path / filename
        spectrogram.save(path, format='png')
        filenames.append(filename)

    return filenames


def create_blast_training_set(stream: Stream):
    # find the start and end of the blast train using sta/lta

    blast_start_time, blast_end_time = window_signal(st)

    end_time = blast_start_time + 0.5
    ct = 1
    while end_time <= blast_end_time:
        create_spectrograms(st, 'blast', end_time)
        end_time += 1


def create_seismic_event_training_set(stream: Stream, event: Event):

    p_pick_time = None
    for arrival in event.preferred_origin().arrivals:
        pick = arrival.get_pick()
        if pick.site == stream[0].stats.site:
            if arrival.phase.upper() == 'P':
                p_pick_time = pick.time

    if p_pick_time is None:
        tt_grid = classifier_project.travel_times.select(
            seed_labels=[stream[0].stats.site], phase='P')
        p_pick_time = tt_grid.interpolate(event.preferred_origin().loc)

    # estimating the event duration
    # measuring the background energy

    background = stream.copy().trim(starttime=p_pick_time-0.5, endtime=p_pick_time-0.1)
    background_energy = np.mean(background[0].data ** 2)

    signal = stream.copy().trim(starttime=p_pick_time).detrend('demean').detrend(
        'linear')[0].data

    energy_window = 100  # sample
    for i in range(len(signal) - energy_window):
        energy_remaining = np.mean(signal[i:i+energy_window] ** 2)
        if energy_remaining < 10 * background_energy:
            end_signal_t = p_pick_time + i / stream[0].stats.sampling_rate
            break

    start_signal = p_pick_time - 0.01
    end_signal = end_signal_t
    duration = end_signal - start_signal
    min_start_window = start_signal + 0.3 * duration - np.min(window_length_seconds)
    max_end_window = end_signal - 0.3 * duration + np.min(window_length_seconds)
    window_range = max_end_window - min_start_window

    event_type = event_type_lookup[event.event_type]

    # from ipdb import set_trace
    # set_trace()
    end_signals = [max_end_window - t for t in (np.random.rand(5) * window_range)]
    # from ipdb import set_trace
    # set_trace()
    # for end in end_signals:

    traces = []
    for tr in stream:
        nsample_end = (end_signal - tr.stats.starttime) * tr.stats.sampling_rate
        tr.stats.sampling_rate = 600
        end_signal = tr.stats.starttime + nsample_end / 600
        traces.append(tr)

    stream.traces = traces

    for perturbation in np.random.rand(5):
        end_signal += perturbation * duration * np.random.choice([-1, 1])
        create_spectrograms(stream, event_type, end_signal)


for context_trace in raw_data_path.glob('*.context_mseed'):
    try:
        st = read(context_trace).detrend('demean').detrend('linear')
    except Exception as e:
        logger.error(e)
    event = read_events(context_trace.with_suffix('.xml'))[0]
    if np.sum(st.composite()[0].data) / len(st.composite()[0].data) > 0.5:
        continue
    # if event.preferred_origin().time < UTCDateTime(2019, 1, 1):
    #     continue

    # if event_type_lookup[event.event_type] == 'blast':
    #     create_blast_training_set(st)

    event_time = event.preferred_origin().time

    st_event = st.copy().trim(starttime=event_time, endtime=event_time + 1)

    if event_type_lookup[event.event_type] == 'seismic event':
        traces = []
        for tr, tr_event in zip(st, st_event):
            if np.std(tr_event.data) > 0.1 * np.max(tr_event.data):
                continue
            traces.append(tr)
        if len(traces) == 0:
            continue
        st.traces = traces
        create_seismic_event_training_set(st, event)

