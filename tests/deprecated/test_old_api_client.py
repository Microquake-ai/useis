from useis.clients.old_api_client.api_client import get_catalog
from pathlib import Path
import tqdm

api_base_url = 'https://api.microquake.org/api/v1/'

output_file_path = Path('data/')
output_file_path.mkdir(parents=True, exist_ok=True)

seismic_events = get_catalog(api_base_url, event_type='seismic event',
                             evaluation_status='accepted', page_size=1000)

other_events = get_catalog(api_base_url, evaluation_status='rejected',
                           page_size=1000)

events = []

for event in seismic_events:
    events.append(event)

for event in other_events:
    events.append(event)

for event in tqdm(events):
    cat = event.get_event()
    cat.write(output_file_path / Path(event.event_file).name)

    fixed_length = event.get_waveforms()
    fixed_length.write(output_file_path / Path(event.waveform_file).name)

    variable_length = event.get_variable_length_waveforms()
    variable_length.write(output_file_path /
                          Path(event.variable_size_waveform_file).name)

    context = event.get_context_waveforms()
    context.write(output_file_path / Path(event.waveform_context_file).name)
