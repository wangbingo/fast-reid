import os, random, shutil
import glob
# from shutil import copyfile
from IPython import embed

# You only need to change this line to your dataset download path
src_path = '../image_A'

if not os.path.isdir(src_path):
    print('please change the src_path')

dst_ds_path = './datasets/pclreid'
if not os.path.isdir(dst_ds_path):
    os.mkdir(dst_ds_path)

#---------------------------------------
#test
query_save_path = dst_ds_path + '/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

gallery_save_path = dst_ds_path + '/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

files = glob.iglob(src_path + '/query/' "*.png")
for file in files:
    cls_name = str(random.randint(10,9999))
    dst_filename = cls_name + '_c' + str(random.randint(1,99)) + 's1_' + file
    embed()


""" 

c = 0

for line in lines:
    line = line.strip()   # to erase blank
    line = line.strip('\n')  # to erase \n
    line_list = line.split(':')
    
    src_img_filename  = line_list[0]
    cls_name          = line_list[1] 
    
    src_file_path = src_path + '/images/' + src_img_filename

    dst_filename = cls_name + '_c' + str(random.randint(1,99)) + 's1_' + src_img_filename
    # dst_filename = cls_name + '_c1s1_' + src_img_filename
    dst_file_path = train_save_path + '/' + dst_filename
    
    shutil.move(src_file_path, dst_file_path)
    
    c += 1
    if c % 10000 == 0:
        print('{} files copied.'.format(c)) 

print('train dataset completed.  {} files copied.'.format(c)) """


