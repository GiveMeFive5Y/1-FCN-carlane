import json

# 数据集中包含的10个类别
categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']

# 图片的分辨率
picture_width = 1282
picture_height = 720


def parseJson(jsonFile):
    '''
      params:
        jsonFile -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，存储了一个json文件里面的方框坐标及其所属的类，
    '''
    objs = []
    obj = []
    info = jsonFile
    name = info['name']
    objects = info['labels']
    for i in objects:
        if (i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj.append(i['category'])
            objs.append(obj)
            obj = []
    return name, objs


# test
file_handle = open('traindata.txt', mode='a')
f = open("/home/violet/Documents/dataset/bdd100k/label/train/bdd100k_labels_images_train.json")  # json文件的绝对路径，换成自己的
info = json.load(f)
objects = info
n = len(objects)


# 将左上右下坐标转换成 中心x,y以及w  h
def bboxtrans(box_x_min, box_y_min, box_x_max, box_y_max):
    x_center = (box_x_min + box_x_max) / (2 * picture_width)
    y_center = (box_y_min + box_y_max) / (2 * picture_height)
    width = (box_x_max - box_x_min) / (2 * picture_width)
    height = (box_y_max - box_y_min) / (2 * picture_height)
    return x_center, y_center, width, height


for i in range(n):
    an = ""
    name, result = parseJson(objects[i])
    an = "./data/custom/images/train/" + name  # 这里我改成了图片的相对路径
    for j in range(len(result)):
        cls_id = categorys.index(result[j][4])
        x, y, w, h = bboxtrans(result[j][0], result[j][1], result[j][2], result[j][3])
        an = an + ' ' + str(cls_id) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
    an = an + '\n'
    file_handle.write(an)
    print(len(result))
    print(an)

#数据整合code
with open('traindata.txt') as f:
    line = f.readline()
    while (line):
        data = line.strip().split()  # strip(): 删除前后空格、空行
        wf = open('./data/' + data[0][27:-4]+'.txt', 'a+')
        numofline = len(data) - 1
        for i in range(1, numofline, 5):
            for j in range(0, 5):
                idx = i + j
                wf.write(data[idx])  # class_index center_x cnter_y w h
                wf.write(' ')
            wf.write('\n')
        line = f.readline()

#提取模型
'''在一个新的python脚本文件中'''
import tensorflow as tf
'''导入其他库'''
pass

'''其他数据准备工作'''
'''这里不需要重新搭建模型'''

'''提取模型，首先提取计算图，这一步相当于搭建模型'''
saver = tf.train.import_meta_graph("model/mnist.ann-10000.meta")

with tf.Session() as sess:
    '''提取保存好的模型参数'''
    '''这里注意模型参数文件名要丢弃后缀.data-00000-of-00001'''
    saver.restore(sess, "model/mnist.ann-10000")

    '''通过张量名获取张量'''
    '''这里按张量名获取了我保存的一个模型的三个张量，并换上新的名字'''
    new_x = tf.get_default_graph().get_tensor_by_name("x:0")
    new_y = tf.get_default_graph().get_tensor_by_name("y:0")
    new_y_ = tf.get_default_graph().get_tensor_by_name("y_:0")
    '''现在可以进行计算了'''
    y_1 = sess.run(new_y_, feed_dict={new_x: new_x_data, new_y: new_y_data})

print(y_1)