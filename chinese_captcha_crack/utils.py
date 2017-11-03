import numpy as np

from cfg import MAX_CAPTCHA, CHAR_SET_LEN,gen_char_set
from PIL import Image


def char2pos(c):
    """
    字符验证码，字符串转成位置信息
    :param c:
    :return:
    """

    return gen_char_set.index(c)


    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('No Map')
    return k


def pos2char(char_idx):
    """
    根据位置信息转化为索引信息
    :param char_idx:
    :return:
    """
    #
    # if not isinstance(char_idx, int64):
    #     raise ValueError('error')

    return gen_char_set[char_idx]

    if char_idx < 10:
        char_code = char_idx + ord('0')
    elif char_idx < 36:
        char_code = char_idx - 10 + ord('A')
    elif char_idx < 62:
        char_code = char_idx - 36 + ord('a')
    elif char_idx == 62:
        char_code = ord('_')
    else:
        raise ValueError('error')

    return chr(char_code)


def convert2gray(img):
    """
    把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    :param img:
    :return:
    """
    img=Image.fromarray(img)
    img=img.convert('L')
    img=np.array(img)
    return img



def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长MAX_CAPTCHA个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN

        char_code = pos2char(char_idx)

        text.append(char_code)
    return "".join(text)


if __name__ == '__main__':
    text = 'XD8K'
    print(text2vec(text))