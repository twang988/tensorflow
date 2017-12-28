import random
from os import path,listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageFilter
#from captcha.image import ImageCaptcha  # pip install captcha
from captcha_modifyed.image import ImageCaptcha
from cfg import MAX_CAPTCHA,IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN
from cfg import gen_char_set,number,ALPHABET,CH_CHAR,train_pic_path
from realdata import pic_convert

# 验证码一般都无视大小写；验证码长度4个字符
WHITE=0
WHITE_THRES=0
def choice_set(if_no_chinese):
    if if_no_chinese==1:
        set_choice=random.randint(1,2)
    else:
        set_choice=random.randint(1,5)
    if set_choice==1 :
        return number
    elif set_choice==2:
        return ALPHABET
    else:
        return CH_CHAR
def random_captcha_text(captcha_size=MAX_CAPTCHA):

    _size=captcha_size#random.randint(1,captcha_size)
    captcha_text = []
    is_no_chinese=random.randint(0,1)
    for i in range(_size):
        c = random.choice(choice_set(1))
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
_is_inited=False
r_path_list=[]
def _init_pic_list():
    global _is_inited,r_path_list
    if _is_inited:
        return r_path_list
    _is_inited=True
    r_path_list=listdir(train_pic_path)
    return r_path_list

def real_captcha_text_and_image():

    path_list=_init_pic_list()
    pic_name=random.choice(path_list)
    text,_=pic_name.split('_')
    pic_full_path = train_pic_path + pic_name
    captcha_image=pic_convert(pic_full_path,None,isfromfile=True)
    captcha_image = np.array(captcha_image)
    return text,captcha_image
def gen_captcha_text_and_image():
    """
    生成字符对应的验证码
    :return:
    """
    path='./font/'
    path2='./font2/'
    all_font=listdir(path)
    font_list=[]
    font_list2=[]
    for i in all_font:
        font_list.append(path+i)
    all_font=listdir(path2)
    for i in all_font:
        font_list2.append(path2+i)
    image = ImageCaptcha(width=200,height=60,fonts=font_list,fonts2=font_list2)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    str=captcha_text.replace('_','')
    captcha = image.generate(str)

    captcha_image = Image.open(captcha)
    captcha_image=captcha_image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
    captcha_image=captcha_image.convert('L')
    global WHITE,WHITE_THRES
    WHITE_THRES=random.randint(70,120)
    WHITE=random.randint(210,255)
    captcha_image=captcha_image.point(_inittable())
    captcha_image = captcha_image.filter(ImageFilter.SMOOTH)

    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image



import requests,base64
headers_getcapcha = {
'Accept':'application/json',
}
payload_getcapcha={}
def captcha_from_web_service(number=1):
    payload_getcapcha['count']=str(number)
    response = requests.get('http://localhost:8080/api/captcha',params=payload_getcapcha,headers=headers_getcapcha)
    text_list=[]
    img_list=[]
    if number==1:
        response=eval(response.text)[0]
        imgdata=base64.b64decode(response['image'])
        captcha_image=pic_convert(imgdata,None,isfromfile=False)
        captcha_image = np.array(captcha_image)

        r1,r2=response['value'].split('_')
        text=r1[1:]+r2[1:]

        return text,captcha_image
    elif number>1:

        response=eval(response.text)
        for i in response:
            imgdata=base64.b64decode(i['image'])
            captcha_image=pic_convert(imgdata,None,isfromfile=False)
            captcha_image = np.array(captcha_image)
            img_list.append(captcha_image)
            r1,r2=i['value'].split('_')
            text=r1[1:]+r2[1:]
            text_list.append(text)
        return text_list,img_list

def wrap_gen_captcha_text_and_image(number=1):
    r=3#random.randint(1,10)
    if r==1:
        text, image = gen_captcha_text_and_image()
    elif r==2:
        text, image =real_captcha_text_and_image()
    elif r==3:
        text, image =captcha_from_web_service(number)
    #
    return text, image


def __gen_and_save_image():
    """
    可以批量生成验证图片集，并保存到本地，方便做本地的实验
    :return:
    """

    for i in range(50000):
        text, image = wrap_gen_captcha_text_and_image()

        im = Image.fromarray(image)

        uuid = uuid.uuid1().hex
        image_name = '__%s__%s.png' % (text, uuid)

        img_root = join(capt.cfg.workspace, 'train')
        image_file = path.join(img_root, image_name)
        im.save(image_file)


def __demo_show_img():
    """
    使用matplotlib来显示生成的图片
    :return:
    """
    text, image = wrap_gen_captcha_text_and_image()

    print("验证码图像channel:", image.shape)  # (60, 160, 3)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


if __name__ == '__main__':
    # gen_and_save_image()
    pass
