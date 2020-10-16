# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:55:34 2020

@author: mouse
"""
import os
def pacs_txt_preprocess(txt_file):
    fread = open(txt_file,'r')
    output_file = txt_file.split('\\')[-1]
    fwrite = open(output_file,'w') 
    line = fread.readline()
    while line:
        line = line.split('/')[3:]
        line = '/'.join(line)
        fwrite.writelines(line)
        line = fread.readline()
    fread.close()
    fwrite.close()
    return

if __name__ == '__main__':
    global_path = r'C:\Users\Dell\Desktop\txt_lists'
    dirlist = os.listdir(global_path)
    for d in dirlist:
        path = global_path + '\\' + d
        pacs_txt_preprocess(path)