# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:09:02 2019

@author: zbr
"""
import os
import csv 
import random

random_times = 10
input_flag = 1

train_str = ''
test_str  = ''

def open_csv(file_name,up_dir_PIC_str):
    with open(file_name,'rt') as myFile:
        lines=csv.reader(myFile)
        for line in lines: 
            if str(line[0]).split(' ') == str(up_dir_PIC_str).split(' '):
                return line[1],line[2],line[3] # return x y z 

def write_str_file(file_name,str_c):
    w_path = file_name
    f = open(w_path,'a')
    f.writelines(str_c)
    f.close()

def record_file_str(dir_tmp,test_flag):
    global train_str
    global test_str
    for root, dirs, files in os.walk(dir_tmp):
        for name in files:
            up_dir = root.split('/')[-1]               #   example: PIC_20190601_185632
            up_up_dir = root.split('/')[-2]            #   example: scene1_jiading_lib_training
            if(name.endswith('.jpg') and up_up_dir[:5]):
                up_dir = root.split('/')[-1]           #   example: PIC_20190601_185632
                up_up_dir = root.split('/')[-2]        #   example: scene8_jiading_hualou_training
                up_up_up_dir = root.split('/')[-3]     #   example: undist_train 
                
                up_path    = os.path.dirname(root)      #   example: ./undist_train/scene8_jiading_hualou_training
                up_up_path = os.path.dirname(os.path.dirname(root))  # example: ./undist_train
                
                csv_path = os.path.join( up_path , up_up_dir+'_coordinates.csv')
                x,y,z = 0,0,0#open_csv(csv_path,up_dir)
                image_path = os.path.join(os.path.join(os.path.join(up_up_path,up_up_dir),up_dir),name)
                up_dir_end_num = up_dir.split('_')[-1]
                if(test_flag):
                    #test_str = test_str + image_path + ' ' + up_dir_end_num + ' ' +str(x)+ ' ' +str(y)+ ' ' +str(z)+'\n'
                    test_str = test_str + image_path+'\n'
                else:
                    train_str = train_str + image_path + '\n'


file_str = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
cur_dir = os.path.dirname(file_str) 

cur_dir_undis_train = os.path.join(cur_dir,'undist_train')#change dirs



if input_flag == 0:
    i_num = 0
else:
    i_num = 3

secen_num_check = 0

secen_num = int(random.random()*10)%8+1

for i in range(i_num):
    secen_num_s = str(input('input the secen num 1-8 --> ')  )
    if(secen_num_s.isdigit()):
        secen_num = int(secen_num_s)
        if(secen_num >=1 and secen_num <=8 ):
            secen_num_check = 1
            break
    else:
        print('please input the number 1-8!, like : 1')
        
if(secen_num_check == 0 and input_flag != 0):
    raise ValueError('please input the number 1-8!, like : 1')  
else:
    print('secen',secen_num)
    print('running..')
    

# extract multi times

for ri in range(random_times):    

    # get some scene pic rand number 0-len(file_pic_list)
    for file_scene in os.listdir(cur_dir_undis_train): # scene1_jiading_lib_training scene2_siping_lib_training ...
        file_pic_list = os.listdir(os.path.join(cur_dir_undis_train,file_scene))
        file_pic_list_l = len(file_pic_list)
        train_str = ''
        test_str  = ''
        if file_scene[5] == str(secen_num):
            pic_num_rand = int(random.random()*10)%file_pic_list_l+1
            for file_pic_index in range(file_pic_list_l):
                file_pic_path = os.path.join(os.path.join(cur_dir_undis_train,file_scene),file_pic_list[file_pic_index])
                if(file_pic_index == pic_num_rand): # test 
                    record_file_str(file_pic_path,1)
                    test_str  = test_str
                #else:
                #    record_file_str(file_pic_path,0)
            write_str_file(  os.path.join(cur_dir,file_scene+'_test.txt' )  ,test_str)
            #write_str_file(  os.path.join(cur_dir,file_scene+'_train.txt' ) ,train_str)
            
        #else:
            #for file_pic_index in range(file_pic_list_l):
            #    file_pic_path = os.path.join(os.path.join(cur_dir_undis_train,file_scene),file_pic_list[file_pic_index])
            #    record_file_str(file_pic_path,0)
            #write_str_file(  os.path.join(cur_dir,file_scene+'_test.txt' )  ,test_str)
            #write_str_file(  os.path.join(cur_dir,file_scene+'_train.txt' ) ,train_str)



                
