import yaml
from urllib.request import urlopen

url_syst = "https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/systematics.yaml"

syst_dict = yaml.load(
    urlopen(url_syst), yaml.FullLoader
    )

def select_systematics(name, keywords):
    if keywords:
        for kw in keywords:
            if kw in name:
                return True
        return False
    else:
        return True

# A helper function that returns a list of systematic uncertainty names
def get_systematics(
    name_filters = [], # list of str; Strings for matching and selecting systematic uncertainties. If empty, take all that are available
    syst_type = None, # str; Required systematic uncertainty type e.g. 'Branch' or 'ScaleFactor'. No requirement if None
    list_of_tuples = False, # bool; If True, return a list of tuples that groups the variations of the same systematic uncertainty together: [(syst1_up,syst1_down), (syst2_up, syst2_down), ...]; Otherwise, return a list of strings
    get_weight_types = False, # bool; If True, also return the associated list of weight types
    ):

    if isinstance(name_filters, str):
        name_filters = [name_filters]

    syst_list = []
    wtype_list = []

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

                    systs, wtypes = [], []
                    for v in variations:
                        syst_var = f"{syst_name}_{v}"

                        if not select_systematics(syst_var, name_filters):
                            continue

                        wtype_var = f"weight_{prefix}_{sname}_{v}:{i}" if stype=="ScaleFactor" else "nominal"

                        systs.append(syst_var)
                        wtypes.append(wtype_var)

                    if list_of_tuples and systs:
                        syst_list.append(tuple(systs))
                        wtype_list.append(tuple(wtypes))
                    else:
                        syst_list += systs
                        wtype_list += wtypes
            else:
                syst_name = f"{prefix}_{s}" if s else f"{prefix}"

                systs, wtypes = [], []
                for v in variations:
                    syst_var = f"{syst_name}_{v}"

                    if not select_systematics(syst_var, name_filters):
                        continue

                    wtype_var = f"weight_{syst_name}_{v}" if stype=="ScaleFactor" else "nominal"

                    systs.append(syst_var)
                    wtypes.append(wtype_var)

                if list_of_tuples and systs:
                    syst_list.append(tuple(systs))
                    wtype_list.append(tuple(wtypes))
                else:
                    syst_list += systs
                    wtype_list += wtypes

    if get_weight_types:
        return syst_list, wtype_list
    else:
        return syst_list

# systematic uncertainty groups
syst_groups = {
    "JES" : {
        "label" : "JES/JER",
        "filters" : ["CategoryReduction_JET_", "weight_jvt"],
    },
    "BTag" : {
        "label" : "Flavor Tagging",
        "filters" : [],
    },
    "Lepton" : {
        "label" : "Lepton",
        "filters" : [],
    },
    "MET" : {
        "label" : "$E_{\text{T}}^{\text{miss}}$",
    },
    "Background" : {
        "label" : "Background"
    },
    "Pileup" : {
        "label" : "Pileup",
        "filters": ["pileup_UP", "pileup_DOWN"],
    },
    "IFSR" : {
        "label" : "IFSR",
    },
    "PDF" : {
        "label" : "PDF",
    },
    "mc_stat": {
        "label" : "MC Stat.",
    },
    "MTop" : {
        "label" : "$m_{\text{t}}$",
    },
    "Hadronization" : {
        "label" : "Hadronization"
    },
    "Unfolding" : {
        "label" : "Unfolding"
    },
    "generator" : {

    }
}