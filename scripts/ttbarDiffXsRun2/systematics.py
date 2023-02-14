import yaml
from urllib.request import urlopen

url_syst = "https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/systematics.yaml"

syst_dict = yaml.load(
    urlopen(url_syst), yaml.FullLoader
    )

# A helper function that returns a list of string tuple:
# [(syst1_up,syst1_down), (syst2_up, syst2_down), ...]
def get_systematics(
    name_filters = [], # list of str; Strings for matching and selecting systematic uncertainties. If empty, take all that are available
    syst_type = None, # str; Required systematic uncertainty type e.g. 'Branch' or 'ScaleFactor'. No requirement if None
    ):

    if isinstance(name_filters, str):
        name_filters = [name_filters]

    syst_list = []

    def select_systematics(name, keywords):
        if keywords:
            for kw in keywords:
                if kw in name:
                    return True
            return False
        else:
            return True

    # loop over syst_dict
    for k in syst_dict:
        stype = syst_dict[k]['type']
        prefix = syst_dict[k]['prefix']
        uncertainties = syst_dict[k].get('uncertainties', [""])
        variations = syst_dict[k]['variations']

        if syst_type is not None and stype != syst_type:
            continue

        for s in uncertainties:

            if isinstance(s, dict):
                # e.g. {'eigenvars_B': 9}
                # A vector of uncertainties
                assert(len(s)==1)
                sname, vector_length = list(s.items())[0]

                for i in range(vector_length):
                    syst_name = f"{prefix}_{sname}{i+1}"

                    if select_systematics(syst_name, name_filters):
                        syst_list.append(tuple(f"{syst_name}_{v}" for v in variations))
            else:
                syst_name = f"{prefix}_{s}" if s else f"{prefix}"

                if select_systematics(syst_name, name_filters):
                    syst_list.append(tuple(f"{syst_name}_{v}" for v in variations))

    return syst_list