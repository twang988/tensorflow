"""
数据生成器
"""

import numpy as np

from cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA
from gen_captcha import wrap_gen_captcha_text_and_image
from utils import convert2gray, text2vec
from PIL import Image

def get_next_batch(batch_size=128):
    """
    # 生成一个训练batch
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        #image = convert2gray(image)

        if False:
            image_=Image.fromarray(image)
            image_.save('./test_out/%s.jpg'%text)
            #exit()

        #image=np.array(image)
        batch_x[i, :] = image.flatten()/255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

def get_next_batch_from_web(batch_size=64):

    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])


    text, image = wrap_gen_captcha_text_and_image(batch_size)
    for i in range(batch_size):
        batch_x[i, :] = image[i].flatten()/255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text[i])

    return batch_x, batch_y