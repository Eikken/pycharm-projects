#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   testcase1.py    
@Time    :   2022/11/1 9:44  
@E-mail  :   iamwxyoung@qq.com
@Tips    :
'''


# 试题 1
def bin2hex(s_='100101100101000'):
    hex_value = hex(int(s_, 2))
    for i in range(1, len(hex_value) // 2):  # 分字节
        reverse_value = hex_value[-2*i+1: -2*i-1: -1]
        print(reverse_value[::-1], end='')
    print()


bin2hex()


# 试题 2
class DeviceInfo:
    def __init__(self, txt_data):
        self.txt_data = txt_data

    def get_dev_name(self):
        name_list = []
        first_split = self.txt_data.split('[')
        for i in first_split:
            this_name = i.split(']')[0]
            name_list.append(this_name)
        return name_list

    def get_dev_id(self, dev_name):

        for i in list(self.txt_data.split('\n')):
            if dev_name in i:
                this_id = i.split('(')[1].split(')')[0]
                return this_id
            return None

    def get_dev_ch_val(self, dev_name, channel_name):
        for i in list(self.txt_data.split('\n')):
            if dev_name in i:
                this_channel = i.split('{')[1].split('}')[0]
                for j in this_channel.split(','):
                    if channel_name in j:
                        return j.split(':')[1]
            return None


txt = '''[ABC](M1P2345){vb_out:1, ld_out:2}
[ADC](M1P2233){vu_out:1}'''

info = DeviceInfo(txt_data=txt)
print(info.get_dev_id(dev_name='ABC'))
print(info.get_dev_ch_val(dev_name='ABC', channel_name='vbout'))