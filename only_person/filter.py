# import glob
#
# for imgid in glob.glob('../../datasets/coco128/images/train2017/*'):
#     print(imgid)
#
# for labelid in glob.glob('../../datasets/coco128/labels/train2017/*'):
#     print(labelid)
#
import os
# dirs1 = '../../datasets/coco128/labels/train2017_person'
# if not os.path.exists(dirs1):
#     os.makedirs(dirs1)
#
# for root,dirs,files in os.walk(r"../../datasets/coco128/labels/train2017"):
#         for file in files:
#             print(root)
#             print(file)
#             openfile = os.path.join(root, file)
#             with open(openfile, 'r') as r:
#                 lines = r.readlines()
#             openfile_person = os.path.join(dirs1, file)
#             with open(openfile_person, 'w') as w:
#                 for l in lines:
#                     class1 = l.strip().split(' ')[0]
#                     if class1 == '0':
#                         w.write(l)


for root, dirs, files in os.walk(r'C:\Users\jiangchenxi\Pictures\depth'):
    for i, file in enumerate(files):
            openfile = os.path.join(root, file)

            newfile = "Part{}_{}".format(i%4+1,file)
            newfile = os.path.join(root, newfile)
            print(openfile)
            print(newfile)
            os.rename(openfile, newfile)
