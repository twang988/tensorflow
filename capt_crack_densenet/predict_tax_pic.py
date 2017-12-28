"""
专门做预测的
"""
import time,os

import numpy as np
import tensorflow as tf

from cfg import MAX_CAPTCHA, CHAR_SET_LEN, model_path
from cnn_sys import crack_captcha_cnn, X, keep_prob
from gen_captcha import wrap_gen_captcha_text_and_image
from utils import convert2gray, vec2text
from realdata import pic_convert

def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)



def get_test_image(path,name,to_path=None):
    full_path = path + name
    info=name.split('_')

    f=open(full_path,'rb')
    data=f.read()
    image=pic_convert(data,None)

    if to_path!=None:
        image.save('%s/%s.jpg'%(to_path,info[0]))
        #exit()

    captcha_image = np.array(image)

    return 0,info[0],captcha_image,0

def batch_hack_captcha():
    """
    批量生成验证码，然后再批量进行识别
    :return:
    """

    # 定义预测计算图
    global_step = tf.Variable(0, trainable=False)
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(save_model + ".meta")


        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        stime = time.time()

        right_cnt = 0
        pic_dir='./tax-pic/'
        all_image = os.listdir(pic_dir)
        task_cnt = 0#len(all_image)
        for i in all_image:
            color,text, image ,full_name= get_test_image(pic_dir,i)
            if not text.encode('utf-8').isalnum():
                continue
            task_cnt+=1
            image = image.flatten() / 255
            predict_text = hack_function(sess, predict, image)
            predict_text=predict_text.replace('_','')
            if text == predict_text:
                print("----标记: {}  预测: {} 颜色: {}".format(text, full_name,color))
                right_cnt += 1
            else:
                print("标记: {}  预测: {}".format(text, predict_text))


        print('task:', task_cnt, ' cost time:', (time.time() - stime), 's')
        print('right/total-----', right_cnt, '/', task_cnt)
        return right_cnt/task_cnt
def export_test_image():
    pic_dir='./tax-pic/'
    all_image = os.listdir(pic_dir)
    for i in all_image:
        color,text, image ,full_name= get_test_image(pic_dir,i,to_path='./test_out/')
def test_hack_captcha(sess,global_step,output):
    """
    批量生成验证码，然后再批量进行识别
    :return:
    """

    # 定义预测计算图
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    stime = time.time()

    right_cnt = 0
    pic_dir='./tax-pic/'
    all_image = os.listdir(pic_dir)
    task_cnt = 0#len(all_image)
    for i in all_image:
        color,text, image ,_= get_test_image(pic_dir,i)
        #if not text.encode('utf-8').isalnum():
            #continue
        task_cnt+=1
        image = image.flatten() / 255
        predict_text = hack_function(sess, predict, image)
        predict_text=predict_text.replace('_','')
        if text == predict_text:
            print("----标记: {}  预测: {} 颜色: {}".format(text, predict_text,color))
            right_cnt += 1
        else:
            print("标记: {}  预测: {} 颜色: {}".format(text, predict_text,color))

    print('task:', task_cnt, ' cost time:', (time.time() - stime), 's')
    print('right/total-----', right_cnt, '/', task_cnt)
    return right_cnt/task_cnt
if __name__ == '__main__':
    #batch_hack_captcha()
    export_test_image()
    print('end...')
