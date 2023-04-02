from useis.processors import classifier
from fastapi import FastAPI, File
from obspy import read
from io import BytesIO
from importlib import reload
reload(classifier)

app = FastAPI()

project = 'classifier'
network = 'OT'
base_directory = '/data_1/projects/'

ec = classifier.Classifier(base_directory, project, network)

app = FastAPI()


@app.post('/classifier/predict/stream', status_code=201)
async def predict(stream: bytes = File(...)):
    f_in = BytesIO(stream)
    st = read(f_in)

    classifier_output = ec.predict(st)

    print(classifier_output)

    return classifier_output.raw_output

@app.get('/classifier/predict/stream', status_code=201)
async def get_nothing():
    return 'bla bla bla'


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