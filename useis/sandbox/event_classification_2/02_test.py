from useis.processors import classifier
from importlib import reload
from pathlib import Path
from uquake.core import read, read_events
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from uquake.core.stream import Stream
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
project_name = 'classification_2'
network_name = 'test'

ec = classifier.Classifier2('/data_1/projects/', 'classification_2', 'test',
                            sampling_rates=[6000])

records = ec.training_db_manager.filter(categories='blast')

root_dir = Path('/data_1/ot-reprocessed-data/')

filenames = np.unique([record.mseed_file for record in records])
np.random.shuffle(filenames)

for f in filenames:
    filename = root_dir / f
    st = read(filename).detrend('demean')
    indices = np.array(random.sample(range(0, len(st)), np.min([20, len(st)])))
    traces = []
    for i in indices:
        traces.append(st[i])

    st = Stream(traces=traces)

    start_time = st[0].stats.starttime
    end_time = start_time + 1
    cat = read_events(filename.with_suffix('.xml'))
    cr = ec.predict(st, event_location=cat[0].origins[-1].loc)
    print(cr)
    # print(cr.predicted_class_ensemble(cat[0].origins[-1].loc))
    plot_results(cr, attention_end=end_time)