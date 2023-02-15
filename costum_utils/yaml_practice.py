import yaml
from yaml import BaseLoader,SafeLoader,UnsafeLoader,FullLoader


import yaml

def constructor(a, b) :
    fields = a.construct_mapping(b)
    return Test(**fields)

yaml.add_constructor('!Test', constructor)

class Test(object) :

    def __init__(self, name, age=30, phone=1100) :
        self.name = name
        self.age = age*20
        self.phone = phone*3

    def __repr__(self):
        return "%s(name=%s, age=%r,phone=%r)" % (self.__class__.__name__, self.name, self.age, self.phone)

print (yaml.load("""
- !Test { name: 'Sam' }
- !Test { name: 'Gaby', age: 20,phone: 5656}""",Loader=FullLoader))



import os
import yaml
import os.path as osp


def get_config(dir='config/config.yaml'):
    # add direction join function when parse the yaml file
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r') as f:
        cfg = yaml.load(f,Loader=UnsafeLoader)

    check_dirs(cfg)

    return cfg


def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(cfg):
    check_dir(cfg['data_root'], mk_dir=False)

    check_dir(cfg['result_root'])
    check_dir(cfg['ckpt_folder'])
    check_dir(cfg['result_sub_folder'])






# with open(r'E:\lxd\OCR_project\text_renderer_lv\costum_utils\yaml_func.yml',mode='r',encoding='utf8') as f:
#     yd = yaml.load_all(f,Loader=SafeLoader)
#     print(list(yd))
