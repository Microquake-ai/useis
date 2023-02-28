import requests
from io import BytesIO, StringIO
from typing import Union
from pathlib import Path
from uquake.core.logging import logger


def get_test_picker_model(output_path: Union[str, Path] = 'test_picker_model.pickle'):
    result = requests.get('https://www.dropbox.com/s/7q2l2uo047rdm73/'
                          'picker_model.pickle?dl=1')
    content = BytesIO(result.content)
    with open(output_path, 'wb') as f_out:
        f_out.write(content.getbuffer())

def get_test_data(output_folder: Union[str, Path]):

    if isinstance(output_folder, str):
        output_folder = Path(output_folder)

    get_context_trace(output_folder)
    get_fixed_length_trace(output_folder)
    get_variable_length_trace(output_folder)
    get_catalog(output_folder)


def get_context_trace(output_folder: Union[str, Path]):
    url = 'https://www.dropbox.com/s/k9xg3cxk7l1f2i7/' \
          '5fc90cac82f5d31fc2acb7290b87f411.context_mseed?dl=1'

    logger.info('extracting context trace')

    result = requests.get(url)
    content = BytesIO(result.content)
    with open(output_folder / f'context.mseed', 'wb') as f_out:
        f_out.write(content.getbuffer())


def get_fixed_length_trace(output_folder: Union[str, Path]):
    url = 'https://www.dropbox.com/s/3co5it512oqu8cy/' \
          '5fc90cac82f5d31fc2acb7290b87f411.mseed?dl=1'

    logger.info('extracting fixed length trace')

    result = requests.get(url)
    content = BytesIO(result.content)
    with open(output_folder / f'fixed_length.mseed', 'wb') as f_out:
        f_out.write(content.getbuffer())


def get_variable_length_trace(output_folder: Union[str, Path]):
    url = 'https://www.dropbox.com/s/dhuq15i5vtvtee8/' \
          '5fc90cac82f5d31fc2acb7290b87f411.'

    logger.info('extracting variable length trace')

    result = requests.get(url)
    content = BytesIO(result.content)
    with open(output_folder / f'variable_length.mseed', 'wb') as f_out:
        f_out.write(content.getbuffer())


def get_catalog(output_folder: Union[str, Path]):
    url = 'https://www.dropbox.com/s/gsh15umzs7r6gaj/' \
          '5fc90cac82f5d31fc2acb7290b87f411.xml?dl=1'

    logger.info('extracting catalogue')

    result = requests.get(url)
    content = BytesIO(result.content)
    with open(output_folder / f'catalogue.xml', 'wb') as f_out:
        f_out.write(content.getbuffer())







# repo = Repo()
# git_url = 'https://jeanphilippemercier/useis_test_data.git'
#
# repo.clone_from(git_url, '.')