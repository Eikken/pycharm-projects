#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   readPDF.py    
@Time    :   2021/3/11 14:08  
@Tips    :   
'''

import os,glob

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
import time

t1 = time.time()
filePath = r'D:\edu\help\代表作引用\文献'
# file_list = os.listdir(filePath)
pdfNameList = glob.glob(filePath+'\\'+'*.pdf')
for i in pdfNameList:
    flag = True
    fp = open(i, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)
    src = PDFResourceManager()
    device = PDFPageAggregator(src,laparams=LAParams())
    inter = PDFPageInterpreter(src,device)
    pages = PDFPage.create_pages(document)
    index = 1
    for page in pages:
        #print(page.contents)
        inter.process_page(page)
        layout = device.get_result()
        print(index)
        index += 1
        for x in layout:
            if isinstance(x, LTTextBoxHorizontal):
                # print(str(x.get_text())) # str类型

                if 'nudged elastic band, NEB' in x.get_text():
                    print(i)
                    flag = False
                    break
    if flag == False:
        break
t = time.time() - t1
print(t)