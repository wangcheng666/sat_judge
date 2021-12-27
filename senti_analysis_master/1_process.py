#!/usr/bin/env python
# -*- coding: utf-8  -*-
#将原始数据合并到一个txt文件

import logging
import os,os.path
import codecs,sys

#读取文件内容
def getContent(fullname):
    f = codecs.open(fullname, 'r', encoding='gb18030',errors='ignore')
    content = f.readlines()
    f.close()
    return content
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])#得到文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    #输入文件目录
    inp = 'data\ChnSentiCorp_htl_ba_2000' 
    folders = ['neg','pos']

    for foldername in folders:
        logger.info("running "+ foldername +" files.")
        
        outp = '2000_' + foldername +'.txt' #输出文件
        output = codecs.open(outp, 'w',encoding='utf8')
        i = 0
        
        rootdir = inp + '\\' + foldername
        #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for parent,dirnames,filenames in os.walk(rootdir):
            for filename in filenames:  #这里并不是按顺序的而是乱序的具体的顺序有系统函数os.walk决定
                f = codecs.open(rootdir + '\\' + filename, 'r', encoding='gb18030',errors='ignore')
                for line in f:
                    
                    # print(rootdir + '\\' + filename)
                    # print(line)
                    #去除空行
                    line = line.strip()  #取出line的开头和结尾的指定字符，默认为空格或换行符即回车
                    if line != '':
                        
                        output.write(line + '\n') #因为strip去除了换行符所以需要加上
                # content = getContent(rootdir + '\\' + filename)
                # output.writelines(content)
                i = i+1
                
        output.close()
        logger.info("Saved "+str(i)+" files.")
                
                
    
    
    
