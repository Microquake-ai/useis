import requests
from io import BytesIO
from obspy import read

project = 'classifier'
network = 'OT'

# Replace this with the path to the mseed file you want to send
mseed_file_path = '/data_1/ot-reprocessed-data/' \
                  '7afae7feef225ae2b05356b8003353ec.context_mseed'


def predict(stream):
    # Create a BytesIO object and write the stream data to it
    stream_io = BytesIO()
    stream.write(stream_io, format='mseed')

    # Reset the buffer position to the beginning of the stream
    stream_io.seek(0)

    # Send the stream to the predict endpoint using requests.post()
    response = requests.post(
        'http://localhost:8000/classifier/predict/stream',
        files={'stream_io': stream_io}
    )

    if response.status_code == 201:
        raw_output = response.json()
        print(raw_output)
    else:
        print(f'Request failed with status code {response.status_code}')

# Read the mseed file into an ObsPy stream object
st = read(open(mseed_file_path, 'rb'))[:12]

# Call the predict function with the ObsPy stream object
response = predict(st)
