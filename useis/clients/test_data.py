import requests
from io import BytesIO

# dropbox_base_url = 'https://www.dropbox.com/sh/kggrtsppg643uef' \
#                    '/AADCMM7RPzQkN6rYmRaw-JWoa/' # ?dl=0'
# 'https://www.dropbox.com/s/7q2l2uo047rdm73/picker_model.pickle?dl=1'

dropbox_suffix = '?dl=1'


def get_test_picker_model(output_path: str = 'test_picker_model.pickle'):
    result = requests.get('https://www.dropbox.com/s/7q2l2uo047rdm73/'
                          'picker_model.pickle?dl=1')
    content = BytesIO(result.content)
    with open(output_path, 'wb') as fout:
        fout.write(content.getbuffer())


repo = Repo()
git_url = 'https://jeanphilippemercier/useis_test_data.git'

repo.clone_from(git_url, '.')