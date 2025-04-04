import os
import yaml
import codecs
import argparse
import omegaconf

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict

_BASE_KEY='_base_'
_INHERIT_KEY = '_inherited_'


def parse_from_yaml(path: str):
    """Parse a yaml file and build config"""
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if _BASE_KEY in dic:
        base_files = dic.pop(_BASE_KEY)
        if isinstance(base_files, str):
            base_files = [base_files]
        for bf in base_files:
            base_path = os.path.join(os.path.dirname(path), bf)
            base_dic = parse_from_yaml(base_path)
            dic = merge_config_dicts(dic, base_dic)

    return dic


def merge_config_dicts(dic, base_dic):
    """Merge dic to base_dic and return base_dic."""
    base_dic = base_dic.copy()
    dic = dic.copy()

    if not dic.get(_INHERIT_KEY, True):
        dic.pop(_INHERIT_KEY)
        return dic

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = merge_config_dicts(val, base_dic[key])
        else:
            base_dic[key] = val

    return base_dic