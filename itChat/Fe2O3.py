#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Fe2O3.py
@Time    :   2020/12/30 11:34
@Tips    :   A x-api-key header, e.g., {‘X-API-KEY’: ‘YOUR_API_KEY’} (recommended method) or
             As a GET (e.g., ?API_KEY=YOUR_API_KEY) or POST variable, e.g., {‘API_KEY’: ‘YOUR_API_KEY’}
             https://www.materialsproject.org/rest/v2/materials/mp-1234/vasp?API_KEY=YOUR_API_KEY
             ues https://materialsproject.org/dashboard to generate a API_KEY; recently one = 'rpFcf1uhxt0nCl3w'
             All URI use https://www.materialsproject.org/rest/v2/{request_type}[/{identifier}][/{parameters}]';
             Use GET https://www.materialsproject.org/rest/v2/materials/{material id, formula, or chemical system}/vasp/{property}
             to generate a new material
             POST https://www.materialsproject.org/rest/v2/query by
             {"criteria": "{'elements':{'$in':['Li', 'Na', 'K'],
              '$all': ['O']}, 'nelements':2}",
              "properties": "['formula',
              'formation_energy_per_atom']"}
            GET or POST api_check https://www.materialsproject.org/rest/v1/api_check
            API 得到数据不但可以单独处理，还可以和我们自己计算出来的数据一起处理。
            比如我们要计算 Li，Fe，O 三种元素的组成的相图，
            我们可以用 mp_entries = m.get_entries_in_chemsys(["Li", "Fe", "O"])得到数据库里有的数据，
            还可以通过pymatgen.apps.borg.hive 里的 VaspToComputedEntryDrone 功能把自己计算的结果和数据库种的数据一起分析：
'''
import pymatgen as mg
from pymatgen.ext.matproj import MPRester
import configparser

def getMyKey():
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    return conf.get('MP_INFO', 'MY_MP_KEY')


MY_MP_KEY = getMyKey()
with MPRester(MY_MP_KEY) as m:
    # structure = m.get_structure_by_material_id('C')
    # dos = m.get_dos_by_material_id('C')
    # bandStructure = m.get_bandstructure_by_material_id('C')
    # #把mp数据库里第1234号结构的，结构数据，dos和band数据得到了，分别保存在structure, dos, bandstructure这三个变量里
    data = m.get_data('Fe2O3')
    # enengies = m.get_data('Fe2O3', 'energy')
    print(data[0])