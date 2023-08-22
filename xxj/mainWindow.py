#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   mainWindow.py    
@Time    :   2022/3/31 16:31  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

from __future__ import unicode_literals
import os

from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PyQt5 import QtGui, QtWidgets, QtCore, QtPrintSupport

# import matplotlib
# matplotlib.use('Qt5Agg')


def imageDecode(*args):
    dat_dir = args[0]
    dat_file_name = args[1]
    target_path = args[2]
    xor_value = int(args[3], 16)

    dat_read = open(dat_dir, "rb")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    out = target_path + "\\" + dat_file_name.split('.')[0] + ".png"
    png_write = open(out, "wb")
    for now in dat_read:
        for nowByte in now:
            newByte = nowByte ^ xor_value
            png_write.write(bytes([newByte]))
    dat_read.close()
    png_write.close()


class ShareInfo:
    showTest = None
    txt = None


class ShowMain:

    def __init__(self):
        self.ui = QUiLoader().load('main.ui')
        self.ui.centralwidget.setStyleSheet("background-color: rgb(	240,248,255); ")
        self.setIcon()
        self.ui.menubar.setStyleSheet("background-color: rgb(255,250,250);"
                                      "QPushButton:pressed{background-color:rgb(0, 206, 209)}")
        self.ui.lbl_tip1.setStyleSheet("font-color:black;font-family:黑体;")
        self.ui.info_wgt.setStyleSheet("background-color: rgb(253,245,230);")
        self.ui.text1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ui.text2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ui.text3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ui.text_area1.setStyleSheet("background-color: rgb(250,250,210);")
        self.ui.btn_exec.setStyleSheet("QPushButton{background-color: rgb(60,179,113); "
                                       "color:white;font-family:黑体;font-size:14;border-radius:8;"
                                       "selection-color:rgb(255,218,185);}"
                                       "QPushButton:pressed{background-color:rgb(0,206,209)}; ")

        self.ui.act_exit.triggered.connect(self.onSignOut)
        self.ui.act_version.triggered.connect(self.version)
        self.ui.btn_exec.clicked.connect(self.executeProgram)

    def executeProgram(self):
        input_value = self.ui.text3.text().strip()  # 每个人不一样
        dat_path = self.ui.text1.text().strip()
        # 修改转换成png图片后的存放路径
        target_path = self.ui.text2.text().strip()
        try:
            fsinfo = os.listdir(dat_path)
            allNum = len(fsinfo)
            count = 1
            for dat_file_name in fsinfo:
                temp_path = os.path.join(dat_path, dat_file_name)
                if not os.path.isdir(temp_path):
                    self.ui.text_area1.setText('共%d个文件，已完成%.2f %%。' % (allNum, count * 100 / allNum))
                    count += 1
                    imageDecode(temp_path, dat_file_name, target_path, input_value)
                else:
                    pass
            self.ui.text_area1.setText('共%d个文件，已完成%.2f %%。' % (allNum, count * 100 / allNum))
            QMessageBox.about(
                self.ui,
                "提示",
                "解析完成！"
            )
        except:
            QMessageBox.warning(
                self.ui,
                "警告",
                "参数解析不正确！"
            )

    def setIcon(self):
        appIcon = QIcon("icon.ico")
        self.ui.setWindowIcon(appIcon)

    def onSignOut(self):
        self.ui.close()

    def version(self):
        QMessageBox.about(self.ui, "Version",
                          """
版本信息 1.0.0
                        
联系 iamwxyoung@qq.com 获取更多信息 
                          
Copyright 2021 Prof. Yan Jiaxu's group."""
                          )


if __name__ == '__main__':
    app = QApplication([])
    ShareInfo.showTest = ShowMain()
    ShareInfo.showTest.ui.show()
    app.exec_()
