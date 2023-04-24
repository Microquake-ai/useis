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

# classifier_model = model.EventClassifier.load()
# classifier_project.add_model(classifier_model)

records = classifier_project.training_db_manager.filter(categories='blast')
# classifier_project.train()


def plot_results(stream, labels, attention_end):

    # Assuming `streams` is a list of obspy streams and `labels` is a list of labels
    fig, axs = plt.subplots(nrows=len(stream), sharex=True, sharey=True,
                            figsize=(15, 12))

    attention_end = attention_end - stream[0].stats.starttime
    attention_start = attention_end - 1

    # Iterate over the streams and plot each one
    for i, (trtr, label) in enumerate(zip(stream, labels)):
        try:
            axes = axs[i]
        except Exception as e:
            axes = axs
        t = np.arange(len(trtr.data)) / trtr.stats.sampling_rate
        trtr.data = trtr.data / trtr.data.max()
        axes.plot(t, trtr.data, 'k')

        # Add labels and title to the plot
        axes.set_ylabel(f"Trace {i + 1}")
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

        axes.text(0.95, 0.95, label, transform=axes.transAxes,
                    fontsize=10, ha='right', va='top',
                    bbox=dict(facecolor='none', edgecolor='none'))

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


root_dir = Path('/data_1/ot-reprocessed-data/')

filenames = np.unique([record.mseed_file for record in records])

for context_mseed in filenames[121:]:
    # input(context_mseed)
    catalog_file = root_dir / context_mseed.replace('context_mseed', 'xml')
    mseed_file = root_dir / context_mseed.replace('context_mseed', 'mseed')

    cat = read_events(str(catalog_file))
    st = read(str(mseed_file)).select(channel='*Z')
    # st = read(str(mseed_file)).composite()
    st_context = read(root_dir / context_mseed)

    event_loc = cat[0].preferred_origin().loc

    distances = []
    stations = []
    locations = []
    for tr in st:
        sites = classifier_project.inventory.select(station=tr.stats.station,
                                                    location=tr.stats.location).sites
        if len(sites) == 0:
            continue
        site = sites[0]
        distances.append(np.linalg.norm(site.loc - event_loc))
        stations.append(site.station_code)
        locations.append(site.location_code)

    indices = np.argsort(distances)
    distances = np.array(distances)
    stations = np.array(stations)
    locations = np.array(locations)

    sta_loc = [f'{sta}.{loc}' for sta, loc in zip(stations, locations)]

    _, i1 = np.unique(sta_loc, return_index=True)

    indices = indices[i1]

    traces = []
    for index in indices[:10]:
        for tr in st.select(station=stations[index], location=locations[index]).copy():
            # tr.stats.station = f'{int(distances[index]):04d}'
            traces.append(tr)

    if len(traces) == 0:
        continue

    st2 = Stream(traces=traces).detrend('demean').detrend('linear')

    end_time = np.min([tr.stats.starttime for tr in st2]) + 2
    outputs = []
    for end_time_perturbation in [0, 0.5, 1]:
        st3 = st2.copy()
        st3.trim(endtime=end_time - end_time_perturbation).detrend('demean').detrend(
            'linear').taper(max_percentage=0.05)
        output = classifier_project.predict(st3)
        outputs.append(output)
        input(output.predicted_class_ensemble(cat[0].preferred_origin().loc))

        plot_results(st2, output.predicted_classes,
                     attention_end=end_time - end_time_perturbation)

    # end_time = st_context[0].stats.endtime
    # outputs = []
    # for end_time_perturbation in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     st3 = st_context.copy()
    #     st3.trim(endtime=end_time - end_time_perturbation)
    #     output = classifier_project.predict(st3)
    #     outputs.append(output)
    #     print(f'{output.predicted_class_ensemble(cat[0].preferred_origin().loc)}\n'
    #           f'{output.raw_output}\n')
    #
    #     plot_results(st_context, output.predicted_classes,
    #                  attention_end=end_time - end_time_perturbation)




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



