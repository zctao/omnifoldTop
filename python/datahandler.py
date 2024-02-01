import util
from datahandler_toy import DataHandlerToy
from datahandler_root import DataHandlerROOT

class DataHandlerFactory:
    def __init__(self):
        self._handlers = {}

    def register(self, key, handler):
        self._handlers[key] = handler

    def get(self, key, *args, **kwargs):
        handler = self._handlers.get(key)
        if not handler:
            raise ValueError(f"Unknown data handler: {key}")
        return handler(*args, **kwargs)

dhFactory = DataHandlerFactory()
dhFactory.register('toy', DataHandlerToy)
dhFactory.register('root', DataHandlerROOT)

def getDataHandler(
    filepaths, # list of str
    variables_reco, # list of str
    variables_truth = [], # list of str
    reweighter = None,
    use_toydata = False,
    **kwargs
    ):
    """
    Get and load a datahandler according to the input file type
    """

    #input_ext = util.getFilesExtension(filepaths)

    if use_toydata:
        dh = dhFactory.get('toy')
        dh.load_data(filepaths)

    #elif input_ext == ".root":
    elif ".root" in filepaths[0]:
        # ROOT files
        dh = dhFactory.get("root", filepaths, variables_reco, variables_truth, **kwargs)

    else:
        #raise ValueError(f"No data handler for files with extension {input_ext}")
        raise ValueError(f"No data handler for files e.g. {filepaths[0]}")

    if reweighter is not None:
        # TODO: check if variables required by reweighter are included
        dh.rescale_weights(reweighter=reweighter)

    return dh