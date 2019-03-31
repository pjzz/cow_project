# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:59:16 2019

@author: ohwada
"""

import ftplib
import time
import os
def ftp(filepath,name):
    host="192.168.11.12"
    user="admin"
    password="pjesfs;sn"
    ftp=ftplib.FTP(host)
    ftp.login(user,password)
    fp=open(filepath,"rb")
    ftp.storbinary("STOR array1/share/video/{}.zip".format(name),fp,102400)
    ftp.close()
    fp.close()
if __name__=="__main__":
    while True:
        path="c:/video/buffer/"
        filenames=os.listdir(path)
        try:
            if(len(filenames)>1):
                ftp(path+filenames[0],filenames[0])
                print("send success")
                os.remove(path+filenames[0])
        except Exception:
            continue
        time.sleep(20)
    
    