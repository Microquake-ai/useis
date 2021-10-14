from librosa.feature import melspectrogram
from librosa import amplitude_to_db
from loguru import logger
import numpy as np
from PIL import Image


def librosa_spectrogram(tr, height=256, width=256, return_data=False,
                        db_scale=True):
    """
        Using Librosa mel-spectrogram to obtain the spectrogram
        :param tr: stream trace
        :param height: image hieght
        :param width: image width
        :return: numpy array of spectrogram with height and width dimension
    """
    # data = get_norm_trace(tr)
    data = tr.normalize().data
    signal = data
    # signal = tr.data
    hl = int(signal.shape[0] // (width * 1.1))  # this will cut away 5% from
    # start and end
    # height_2 = height * tr.stats.sampling_rate // max_frequency
    spec = melspectrogram(signal, n_mels=height,
                          hop_length=int(hl))

    # spec = spec / np.max(spec)
    if db_scale:
        spec = amplitude_to_db(spec)
        spec[spec < 0] = 0
    # resolution = 255
    # spec = np.log(spec + np.max(spec)/ resolution)
    spec = spec / np.max(spec) * 255
    img = spec
    start = (img.shape[1] - width) // 2

    spectrogram = img[: , start:start + width]
    if return_data:
        return spec, Image.fromarray(np.uint8(spectrogram), 'L')

    return Image.fromarray(np.uint8(spectrogram), 'L')


#############################################
# Data preparation
#############################################
def get_norm_trace(tr, taper=True):
    """
    :param tr: microquake.core.Stream.trace
    :param taper: Boolean
    :return: normed composite trace
    """

    # c = tr[0]
    # c = tr.composite()
    tr = tr.detrend('demean').detrend('linear').taper(max_percentage=0.05,
                                                      max_length=0.01)
    tr.data = tr.data / np.abs(tr.data).max()

    nan_in_trace = np.any(np.isnan(tr.data))

    if nan_in_trace:
        logger.warning('NaN found in trace. The NaN will be set '
                       'to 0.\nThis may cause instability')
        tr.data = np.nan_to_num(tr.data)

    return tr.data