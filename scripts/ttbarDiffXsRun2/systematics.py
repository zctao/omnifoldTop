import yaml
from urllib.request import urlopen

url_syst = "https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/systematics.yaml"

syst_dict = yaml.load(
    urlopen(url_syst), yaml.FullLoader
    )

syst_dict.update({
    'scale': {
        'type' : 'GenWeight',
        'prefix' : 'scale',
        'uncertainties' : ['muF', 'muR'],
        'variations' : ['up', 'down']
    },
    'isr' : {
        'type' : 'GenWeight',
        'prefix' : 'isr',
        'uncertainties' : ['alphaS'],
        'variations' : ['Var3cUp', 'Var3cDown']
    },
    'fsr' : {
        'type' : 'GenWeight',
        'prefix' : 'fsr',
        'uncertainties' : ['muR'],
        'variations' : ['up', 'down']
    },
    'pdf' : {
        'type' : 'GenWeight',
        'prefix' : 'PDF4LHC15',
        'variations' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    },
    'hdamp' : {
        'type' : 'Modelling',
        'variations' : ['hdamp']
    },
    'mtop' : {
        'type' : 'Modelling',
        'prefix' : 'mtop',
        'variations' : ['mt169', 'mt176']
    },
    'hard_scattering' : {
        'type' : 'Modelling',
        'prefix' : 'generator',
        'variations' : ['amc'] # aMCatNLO
    },
    # TODO alternative generator uncetainty: PP8pthard1, PP8pthard2
    'hadronization' : { # parton showering
        'type' : 'Modelling',
        'prefix' : 'ps',
        'variations' : ['hw'] # PWGH7
    },
    #'lineshape' : {
    #    'type' : 'Modelling',
    #    'prefix' : 'lineshape',
    #    'variations' : ['madspin'] # TODO: PhP8MadSpin
    #},
    #'matching' : {
    #    'type' : 'Modelling',
    #    'prefix' : 'matching',
    #    'variations' : ['pp8pthard'] # TODO: PP8pthard
    #},
    #
    'singleTop_tW' : {
        'type' : 'BackgroundModelling',
        'prefix' : 'singleTop',
        'uncertainties' : ['tW'],
        'variations' : ['DS']
    },
    'singleTop_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'singleTop',
        'uncertainties' : ['norm'],
        'variations' : [1.05],
    },
    #
    'VV_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'VV',
        'uncertainties' : ['norm'],
        'variations' : [1.06]
    },
    'ttV_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'ttV',
        'uncertainties' : ['norm'],
        'variations' : [1.13]
    },
    'Wjets_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'Wjets',
        'uncertainties' : ['norm'],
        'variations' : [1.5]
    },
    'Zjets_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'Zjets',
        'uncertainties' : ['norm'],
        'variations' : [1.5]
    },
    'fakes_norm' : {
        'type' : 'BackgroundNorm',
        'prefix' : 'fakes',
        'uncertainties' : ['norm'],
        'variations' : [1.5]
    },
    'lumi' : {
        'type' : 'Norm',
        'prefix' : 'lumi',
        'variations' : [1.0083]
    },
})

# For now, taken from https://gitlab.cern.ch/ttbarDiffXs13TeV/pyTTbarDiffXs13TeV/-/blob/DM_ljets_resolved/python/MC_variations.py
# Check from the sumWeights file instead?
gen_weights_dict = {
    "nominal":" nominal "," nominal ":0,
    "scale_muF_up":" muR = 1.0, muF = 2.0 "," muR = 1.0, muF = 2.0 ":1,
    "scale_muF_down":" muR = 1.0, muF = 0.5 "," muR = 1.0, muF = 0.5 ":2,
    "scale_muR_up":" muR = 2.0, muF = 1.0 "," muR = 2.0, muF = 1.0 ":3,
    "scale_muR_down":" muR = 0.5, muF = 1.0 "," muR = 0.5, muF = 1.0 ":4,
    "PDF4LHC15_0":" PDF set = 90900 "," PDF set = 90900 ":11,
    "PDF4LHC15_1":" PDF set = 90901 "," PDF set = 90901 ":115,
    "PDF4LHC15_2":" PDF set = 90902 "," PDF set = 90902 ":116,
    "PDF4LHC15_3":" PDF set = 90903 "," PDF set = 90903 ":117,
    "PDF4LHC15_4":" PDF set = 90904 "," PDF set = 90904 ":118,
    "PDF4LHC15_5":" PDF set = 90905 "," PDF set = 90905 ":119,
    "PDF4LHC15_6":" PDF set = 90906 "," PDF set = 90906 ":120,
    "PDF4LHC15_7":" PDF set = 90907 "," PDF set = 90907 ":121,
    "PDF4LHC15_8":" PDF set = 90908 "," PDF set = 90908 ":122,
    "PDF4LHC15_9":" PDF set = 90909 "," PDF set = 90909 ":123,
    "PDF4LHC15_10":" PDF set = 90910 "," PDF set = 90910 ":124,
    "PDF4LHC15_11":" PDF set = 90911 "," PDF set = 90911 ":125,
    "PDF4LHC15_12":" PDF set = 90912 "," PDF set = 90912 ":126,
    "PDF4LHC15_13":" PDF set = 90913 "," PDF set = 90913 ":127,
    "PDF4LHC15_14":" PDF set = 90914 "," PDF set = 90914 ":128,
    "PDF4LHC15_15":" PDF set = 90915 "," PDF set = 90915 ":129,
    "PDF4LHC15_16":" PDF set = 90916 "," PDF set = 90916 ":130,
    "PDF4LHC15_17":" PDF set = 90917 "," PDF set = 90917 ":131,
    "PDF4LHC15_18":" PDF set = 90918 "," PDF set = 90918 ":132,
    "PDF4LHC15_19":" PDF set = 90919 "," PDF set = 90919 ":133,
    "PDF4LHC15_20":" PDF set = 90920 "," PDF set = 90920 ":134,
    "PDF4LHC15_21":" PDF set = 90921 "," PDF set = 90921 ":135,
    "PDF4LHC15_22":" PDF set = 90922 "," PDF set = 90922 ":136,
    "PDF4LHC15_23":" PDF set = 90923 "," PDF set = 90923 ":137,
    "PDF4LHC15_24":" PDF set = 90924 "," PDF set = 90924 ":138,
    "PDF4LHC15_25":" PDF set = 90925 "," PDF set = 90925 ":139,
    "PDF4LHC15_26":" PDF set = 90926 "," PDF set = 90926 ":140,
    "PDF4LHC15_27":" PDF set = 90927 "," PDF set = 90927 ":141,
    "PDF4LHC15_28":" PDF set = 90928 "," PDF set = 90928 ":142,
    "PDF4LHC15_29":" PDF set = 90929 "," PDF set = 90929 ":143,
    "PDF4LHC15_30":" PDF set = 90930 "," PDF set = 90930 ":144,
    "isr_alphaS_Var3cUp":"Var3cUp","Var3cUp":193,
    "isr_alphaS_Var3cDown":"Var3cDown","Var3cDown":194,
    "fsr_muR_up":"isr:muRfac=1.0_fsr:muRfac=2.0","isr:muRfac=1.0_fsr:muRfac=2.0":198,
    "fsr_muR_down":"isr:muRfac=1.0_fsr:muRfac=0.5","isr:muRfac=1.0_fsr:muRfac=0.5":199,
}

def get_gen_weight_index(syst_name):
    if not syst_name in gen_weights_dict:
        raise KeyError(f"systematics: unknown generator weight: {syst_name}")

    weight_name = gen_weights_dict[syst_name]
    weight_index = gen_weights_dict[weight_name]
    return weight_index

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

        prefix = syst_dict[k].get('prefix','')
        if prefix: # add a trailing underscore
            prefix = f"{prefix}_"

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
                    syst_name = f"{prefix}{sname}{i+1}"

                    systs, wtypes = [], []
                    for v in variations:
                        syst_var = f"{syst_name}_{v}"

                        if not select_systematics(syst_var, name_filters):
                            continue

                        wtype_var = "nominal"
                        if stype == "ScaleFactor":
                            wtype_var = f"weight_{prefix}{sname}_{v}:{i}"
                        elif stype == "GenWeight":
                            wtype_var = f"mc_generator_weights:{get_gen_weight_index(syst_var)}"

                        systs.append(syst_var)
                        wtypes.append(wtype_var)

                    if list_of_tuples and systs:
                        syst_list.append(tuple(systs))
                        wtype_list.append(tuple(wtypes))
                    else:
                        syst_list += systs
                        wtype_list += wtypes
            else:
                syst_name = f"{prefix}{s}_" if s else f"{prefix}"

                systs, wtypes = [], []
                for v in variations:
                    syst_var = f"{syst_name}{v}"

                    if not select_systematics(syst_var, name_filters):
                        continue

                    wtype_var = "nominal"
                    if stype == "ScaleFactor":
                        wtype_var = f"weight_{syst_var}"
                    elif stype == "GenWeight":
                        wtype_var = f"mc_generator_weights:{get_gen_weight_index(syst_var)}"

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
        "filters" : ["bTagSF_DL1r_"],
    },
    "Lepton" : {
        "label" : "Lepton",
        "filters" : ["EG_", "MUON_", "leptonSF_"],
    },
    "MET" : {
        "label" : "$E_{\text{T}}^{\text{miss}}$",
        "filters" : ["MET_"],
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
        "filters" : ['scale_mu','isr_','fsr_'],
    },
    "PDF" : {
        "label" : "PDF",
        "filters" : ['PDF4LHC15_']
    },
    "MTop" : {
        "label" : "$m_{\text{t}}$",
        "filters" : ["mtop_"]
    },
    "hdamp" : {
        "label" : "$h_{\text{damp}}$ variation",
        "filters" : ["hdamp"]
    },
    "Hadronization" : {
        "label" : "Hadronization",
        "filters" : ["ps_hw"]
    },
    "Generator" : {
        "label" : "Hard Scattering",
        "filters" : ['generator_amc']
    },
    "Backgrounds" : {
        "label" : "Backgrounds",
        "filters" : ['singleTop_tW', 'singleTop_norm', 'VV_norm', 'ttV_norm', 'Wjets_norm', 'Zjets_norm', 'fakes_norm'],
    },
    "Lumi" : {
        "label" : "Luminosity",
        "filters" : ['lumi']
    },
    "Unfolding" : {
        "label" : "Unfolding",
        "filters" : []
    },
    "MCStat": {
        "label" : "MC Stat.",
    },
    # combined
    "Lepton+MET" : {
        "label": "Lepton, $E_{\text{T}}^{\text{miss}}$",
        "filters" : ["EG_", "MUON_", "leptonSF_", "MET_"],
    },
}