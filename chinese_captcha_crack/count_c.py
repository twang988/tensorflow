import os
pic_dir='./tax-pic/'

f=open('./chinese.txt','r')
CH_CHAR=[]
lines=f.read()
f.close()
CH_CHAR=eval(lines)

def check_if_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    return False


L=[]
def count_chinese():
    all_image = os.listdir(pic_dir)
    for i in all_image:
        name,_,_=i.split('-')
        for c in name:
            if check_if_chinese(c):
                L.append(c)




