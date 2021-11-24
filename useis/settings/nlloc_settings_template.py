from uquake.nlloc.nlloc import (Control, GeographicTransformation,
                                LocSearchOctTree, LocationMethod,
                                GaussianModelErrors, LocQual2Err)

nlloc_control = {'control': Control(message_flag=1),
                 'transformation': GeographicTransformation(),
                 'locsig': 'Organisation Name',
                 'loccom': '',
                 'locsearch': LocSearchOctTree(20, 20, 20, 1e-6, 50000,
                                               1000, 0, 1),
                 'locmeth': LocationMethod('EDT_OT_WT', 9999.0, 4,
                                           -1, -1, -1, 0, 0),
                 'locgau': GaussianModelErrors(1e-2, 1e-2),
                 'locqual2err': LocQual2Err(1e-4, 1e-4, 1e-4, 1e-4,
                                            1e-4)
                 }
