from useis.processors import classifier
from importlib import reload
from pathlib import Path
from uquake.core import read, read_events
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from uquake.core.stream import Stream
from time import time
reload(classifier)


def plot_results(classifier_results, attention_end):

    # labels = classifier_results.raw_output_string
    stream = classifier_results.inputs
    raw_output_strings = classifier_results.raw_output_strings

    # Assuming `streams` is a list of obspy streams and `labels` is a list of labels
    fig, axs = plt.subplots(nrows=len(stream), sharex=True, sharey=True,
                            figsize=(15, 12))

    attention_end = attention_end - stream[0].stats.starttime
    attention_start = attention_end - 1

    stream = stream.copy().filter('highpass', freq=100)
    # Iterate over the streams and plot each one

    for i, (trtr, label) in enumerate(zip(stream, raw_output_strings)):
        try:
            axes = axs[i]
        except Exception as e:
            axes = axs
        t = np.arange(len(trtr.data)) / trtr.stats.sampling_rate
        trtr.data = trtr.data / trtr.data.max()
        axes.plot(t, trtr.data, 'k')

        # Add labels and title to the plot
        axes.set_ylabel(f"{trtr.stats.site}")
        # axes.set_xlabel("Time (s)")
        # axes.set_title(f"{label}\nTrace {i + 1}")

        axes.set_ylim([-1.1, 1.1])

        # Draw a rectangle to identify the attention window
        # attention_start = trace.stats.starttime + 0.5
        # attention_end = attention_start + 1.0
        rect = Rectangle((attention_start, axes.get_ylim()[0]),
                         attention_end - attention_start,
                         axes.get_ylim()[1] - axes.get_ylim()[0],
                         linewidth=1, edgecolor='r', facecolor='r', alpha=0.2)
        axes.add_patch(rect)

        label = label + \
                f' - {trtr.stats.station}.{trtr.stats.location}.{trtr.stats.channel}'

        axes.text(0.95, 0.95, label, transform=axes.transAxes,
                  fontsize=10, ha='right', va='top',
                  bbox=dict(facecolor='gray', edgecolor='black'))

    # Add a common title for the figure
    # plt.subtitle("Multiple Time Series with Attention Windows")
    try:
        axs[-1].set_xlabel("Time (s)")
    except:
        axs.set_xlabel("Time (s)")

    # Adjust the layout and show the plot
    # fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    # plt.xlim([0, 2])
    plt.subplots_adjust(hspace=0)
    plt.show()


root_dir = '/data_1/projects/'
project_name = 'classification_3'
network_name = 'OT'

ec = classifier.Classifier2('/data_1/projects/', 'classification_3', 'OT',
                            gpu=False)

records = ec.training_db_manager.filter(categories='noise')
root_dir = Path('/data_1/ot-reprocessed-data/')
filenames = np.unique([record.mseed_file for record in records])

# filenames = [f for f in Path('/data_1/redlake/project/data/MSEED').glob('*.mseed')]
np.random.shuffle(filenames)

for f in filenames:
    filename = root_dir / f
    st = read(filename).detrend('demean')
    cat = read_events(filename.with_suffix('.xml'))
    st2 = ec.__select_based_on_distance__(st, cat, n_sites=10)
    # indices = np.array(random.sample(range(0, len(st)), np.min([20, len(st)])))
    # traces = []
    # for i in indices:
    #     tr = st[i].copy().filter('highpass', freq=50).detrend('demean').taper(
    #         max_length=0.01, max_percentage=0.1)
    #     # ec.spectrogram(st[i].copy())
    #     if 'FN' in st[i].stats.channel:
    #         tr = tr.integrate()
    #
    #     # tr.resample(6000)
    #     traces.append(st[i])
    #
    #     # tr.stats.sampling_rate /= 3
    #
    # st = Stream(traces=traces)

    start_time = st[0].stats.starttime
    end_time = start_time + 1
    cat = read_events(filename.with_suffix('.xml'))
    t0 = time()
    cr = ec.predict(st2, event_location=cat[0].origins[-1].loc)
    t1 = time()
    # cr = ec.predict(st, cut_from_start=True)
    print(cr)
    input(f'done predicting in {t1 - t0:0.2f} seconds, for {len(st)} traces')
    # print(cr.predicted_class_ensemble(cat[0].origins[-1].loc))
    # cr.inputs.filter('highpass', freq=100)
    plot_results(cr, attention_end=end_time)
