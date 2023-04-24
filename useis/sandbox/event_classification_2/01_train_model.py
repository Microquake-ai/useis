from useis.processors import classifier
from importlib import reload
from pathlib import Path
reload(classifier)

root_dir = '/data_1/projects/'
project_name = 'classification_2'
network_name = 'test'

ec = classifier.Classifier2('/data_1/projects/', 'classification_2', 'test',
                            sampling_rates=[6000])

ec.train(learning_rate=0.001, batch_size=500, nb_epoch=51,
         save_intermediate_models=True, weight_decay=0.001,
         override_current_model=True, dropout_prob=0.2)

# import cProfile
# f = Path('/data_1/ot-reprocessed-data/f7ea5bf15e545fdbb663a2c26f372561.mseed')
# cProfile.run('ec.__process__(f)')

# import cProfile(ec.__process__())