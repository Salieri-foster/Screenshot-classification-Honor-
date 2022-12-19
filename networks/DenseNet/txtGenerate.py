'''
生成训练集和测试集，保存在txt文件中
'''
import os
import random


train_ratio = 0.6


test_ratio = 1-train_ratio

rootdata = r"dataset"

#数组定义
train_list, test_list = [],[]
data_list = []

class_flag = -1
# 将绝对路径加载进数组中
for a,b,c in os.walk(rootdata):#os.walk可以获得根路径、文件夹以及文件，并会一直进行迭代遍历下去，直至只有文件才会结束
    print(a)
    for i in range(len(c)):
        data_list.append(os.path.join(a,c[i]))

    for i in range(0,int(len(c)*train_ratio)):
        train_data = os.path.join(a, c[i])+'\t'+str(class_flag)+'\n' #class_flag表示分类的类别
        train_list.append(train_data)

    for i in range(int(len(c) * train_ratio),len(c)):
        test_data = os.path.join(a, c[i]) + '\t' + str(class_flag)+'\n'
        test_list.append(test_data)

    class_flag += 1

print(train_list)
# 将数组的内容打乱顺序
random.shuffle(train_list)
random.shuffle(test_list)

#分别将绝对路径对应的数组内容写进文本文件里
with open('train.txt','w',encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('test.txt','w',encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(test_img)
