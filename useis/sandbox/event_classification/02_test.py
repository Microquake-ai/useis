from useis.processors import classifier
from importlib import reload
from useis.ai import model
from uquake.core.logging import logger
from tqdm import tqdm
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from uquake import read, read_events
from pathlib import Path
from uquake.core.stream import Stream

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import uquake
reload(classifier)
reload(model)

torch.cuda.empty_cache()

classifier_project = classifier.Classifier('/data_1/projects/', 'classifier', 'OT',
                                           reset_training=False)

records = classifier_project.training_db_manager.filter(categories='noise')
# classifier_project.train()


def plot_results(stream, labels, attention_end, attention_start):

    # Assuming `streams` is a list of obspy streams and `labels` is a list of labels
    fig, axs = plt.subplots(nrows=len(stream), sharex=True, sharey=True,
                            figsize=(15, 12))

    attention_end = attention_end - stream[0].stats.starttime
    attention_start = attention_end - 1

    # Iterate over the streams and plot each one
    for i, (tr, label) in enumerate(zip(stream, labels)):
        t = np.arange(len(tr.data)) / tr.stats.sampling_rate
        tr.data = tr.data / tr.data.max()
        axs[i].plot(t, tr.data, 'k')

        # Add labels and title to the plot
        axs[i].set_ylabel(f"Trace {i + 1}")
        # axs[i].set_xlabel("Time (s)")
        # axs[i].set_title(f"{label}\nTrace {i + 1}")

        axs[i].set_ylim([-1.1, 1.1])

        # Draw a rectangle to identify the attention window
        # attention_start = trace.stats.starttime + 0.5
        # attention_end = attention_start + 1.0
        rect = Rectangle((attention_start, axs[i].get_ylim()[0]),
                         attention_end - attention_start,
                         axs[i].get_ylim()[1] - axs[i].get_ylim()[0],
                         linewidth=1, edgecolor='r', facecolor='r', alpha=0.2)
        axs[i].add_patch(rect)

        axs[i].text(0.95, 0.95, label, transform=axs[i].transAxes,
                    fontsize=10, ha='right', va='top',
                    bbox=dict(facecolor='none', edgecolor='none'))

    # Add a common title for the figure
    # plt.subtitle("Multiple Time Series with Attention Windows")
    axs[-1].set_xlabel("Time (s)")

    # Adjust the layout and show the plot
    # fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.xlim([0, 2])
    plt.subplots_adjust(hspace=0)
    plt.show()


root_dir = Path('/data_1/ot-reprocessed-data/')

for record in records[1:]:
    context_mseed = record.mseed_file
    catalog_file = root_dir / context_mseed.replace('context_mseed', 'xml')
    mseed_file = root_dir / context_mseed.replace('context_mseed', 'mseed')

    cat = read_events(str(catalog_file))
    st = read(str(mseed_file))
    st_context = read(root_dir / record.mseed_file)

    event_loc = cat[0].preferred_origin().loc

    distances = []
    stations = []
    locations = []
    for site in classifier_project.inventory[0].sites:
        distances.append(np.linalg.norm(site.loc - event_loc))
        stations.append(site.station_code)
        locations.append(site.location_code)

    indices = np.argsort(distances)
    distances = np.array(distances)
    stations = np.array(stations)
    locations = np.array(locations)

    traces = []
    for index in indices[:20]:
        for tr in st.select(station=stations[index], location=locations[index]).copy():
            tr.stats.station = f'{int(distances[index]):04d}'
            traces.append(tr)

    st2 = Stream(traces=traces)

    end_time = st2[0].stats.endtime
    outputs = []
    for end_time_perturbation in [0, 0.5, 1]:
        st3 = st2.copy()
        st3.trim(endtime=end_time - end_time_perturbation)
        output = classifier_project.predict(st3)
        outputs.append(output)
        plot_results(st2.composite(), outputs[0].predicted_classes[1::3],
                    attention_end=end_time - end_time_perturbation,
                    attention_start=end_time - end_time_perturbation - 1)




    # # find the traces from the 10 nearest sensors
    # distances = {}
    # for arrival in cat[0].preferred_origin().arrivals:
    #     station_code = arrival.pick.waveform_id.station_code
    #     if station_code in distances.keys():
    #         continue
    #     distances[station_code] = arrival.distance

    # output = classifier_project.predict(st[0:50])

    # krapout




# st = read('/data_1/ot-reprocessed-data/177f51f6d4485d2ab4396ec7b02799f8.mseed')



