import yaml
from urllib.request import urlopen

url_syst = "https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/systematics.yaml"

syst_dict = yaml.load(
    urlopen(url_syst), yaml.FullLoader
    )