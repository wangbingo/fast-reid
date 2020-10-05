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
count_q = 0
for file in files:
    cls_name = str(random.randint(10,9999))
    ori_filename = file.split('/')[-1]
    dst_file_name = cls_name + '_c' + str(random.randint(1,9)) + 's1_' + ori_filename
    dst_file_path = query_save_path + '/' + dst_file_name
    shutil.move(file, dst_file_path)
    count_q += 1
print('query test dataset completed.  {} files copied.'.format(count_q))

files = glob.iglob(src_path + '/gallery/' "*.png")
count_g = 0
for file in files:
    cls_name = str(random.randint(10,9999))
    ori_filename = file.split('/')[-1]
    dst_file_name = cls_name + '_c' + str(random.randint(1,9)) + 's1_' + ori_filename
    dst_file_path = gallery_save_path + '/' + dst_file_name
    shutil.move(file, dst_file_path)
    count_g += 1
print('gallery test dataset completed.  {} files copied.'.format(count_g))








