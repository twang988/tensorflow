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


        if False:#此处改为Ture,用以输出查看训练图片
            image_=Image.fromarray(image)
            image_.save('./test_out/%s.jpg'%text)
            #exit()

        batch_x[i, :] = image.flatten()/255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y
