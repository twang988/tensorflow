import random
from os import path,listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageFilter
#from captcha.image import ImageCaptcha  # pip install captcha
from captcha_modifyed.image import ImageCaptcha#使用修改过的captcha生成器
from cfg import MAX_CAPTCHA,IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN
from cfg import gen_char_set,number,ALPHABET,CH_CHAR



WHITE=0
WHITE_THRES=0
def choice_set(if_no_chinese):
    if if_no_chinese==1:
        set_choice=random.randint(1,2)
    else:
        set_choice=random.randint(1,3)
    if set_choice==1 :
        return number
    elif set_choice==2:
        return ALPHABET
    else:
        return CH_CHAR
def random_captcha_text(captcha_size=MAX_CAPTCHA):

    _size=random.randint(1,captcha_size)
    captcha_text = []
    is_no_chinese=random.randint(0,2)
    for i in range(_size):
        c = random.choice(choice_set(0))
        captcha_text.append(c)
    if len(captcha_text)<MAX_CAPTCHA:
        for i in range(MAX_CAPTCHA-len(captcha_text)):
            captcha_text.append('_')
    return captcha_text

def _inittable():
    table=[]
    global WHITE,WHITE_THRES

    for i in range(256):
        if i>WHITE_THRES:
            table.append(WHITE)
        else:
            table.append(0)
    return table
def gen_captcha_text_and_image():
    """
    生成字符对应的验证码
    :return:
    """
    path='./font/'#英文字体目录
    path2='./font2/'#中文字体目录
    all_font=listdir(path)
    font_list=[]
    font_list2=[]
    for i in all_font:
        font_list.append(path+i)
    all_font=listdir(path2)
    for i in all_font:
        font_list2.append(path2+i)
    image = ImageCaptcha(width=180,height=72,fonts=font_list,fonts2=font_list2)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    str=captcha_text.replace('_','')
    captcha = image.generate(str)

    captcha_image = Image.open(captcha)
    captcha_image=captcha_image.resize((90,36))
    captcha_image=captcha_image.convert('L')
    global WHITE,WHITE_THRES
    WHITE_THRES=random.randint(70,120)
    WHITE=random.randint(210,255)
    captcha_image=captcha_image.point(_inittable())
    captcha_image = captcha_image.filter(ImageFilter.SMOOTH)

    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def wrap_gen_captcha_text_and_image():
    text, image = gen_captcha_text_and_image()
    return text, image


