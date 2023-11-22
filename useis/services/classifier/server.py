from useis.processors import classifier
from fastapi import FastAPI, File
from obspy import read
from io import BytesIO
from importlib import reload
from useis.services.models.classifier import ClassifierResults
reload(classifier)

app = FastAPI()

project = 'classifier'
network = 'OT'
base_directory = 'projects/'

ec = classifier.Classifier(base_directory, project, network, gpu=False)

app = FastAPI()


# Define the endpoint for handling predictions
@app.post('/classifier/predict/stream', status_code=201,
          response_model=ClassifierResults)
async def predict(stream: bytes = File(...)):
    """
    Endpoint for making predictions on a stream of data.

    Args:
        stream (bytes, required): A byte stream of data in miniSEED format.

    Returns:
        A JSON string representing the prediction output. The string will
        contain a list of dictionaries, where each dictionary corresponds
        to a predicted label for a specific time window in the input stream.
    """

    # Read the stream data into a BytesIO object
    f_in = BytesIO(stream)

    # Use ObsPy to read the stream data into a Stream object
    st = read(f_in)

    # Make a prediction using the Stream object and a pre-trained classifier
    classifier_output = ec.predict(st)

    # Print the prediction (optional)
    print(classifier_output)

    # Convert the prediction to a JSON string and return it as the response
    return classifier_output.to_fastapi()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Your API title",
        version="0.1.0",
        description="Your API description",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Your API title",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    return custom_openapi()