import os,base64
import requests
import re,copy
import time,random
from io import BytesIO
import threading
import numpy as np
from PIL import Image,ImageFilter


woker_number=5
save_path='./pics'

payload = {'callback': 'jQuery110205660534614759527_1508391681367',
'fpdm': '1100172320',
'r':'0.1375104796068471',
'_':'None'}


clolr_map={'00':'void','01':'red','02':'yellow','03':'blue'}

headers = {'Connection':'keep-alive',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.91 Safari/537.36',
'Accept':'*/*',
'Referer':'https://inv-veri.chinatax.gov.cn/',
'Accept-Encoding':'gzip, deflate, br',
'Accept-Language':'zh-CN,zh;q=0.8,en;q=0.6',
'Host': 'zjfpcyweb.bjsat.gov.cn'
}

def enhance_color(im,mode):
    width, _ = im.size
    threshold=30
    maxthreshold=160
    minthreshold=126
    COLOR=213
    for i, px in enumerate(im.getdata()):
        y = int(i / width)
        x = i % width
        r,g,b=px
        if mode=='blue':
            if b>maxthreshold and g<119 and r<120:
                #im.putpixel((x, y), (b, b, b))
                #im.putpixel((x, y), (255, 255, 255))
                im.putpixel((x, y), (COLOR, COLOR, COLOR))
            else:
                #im.putpixel((x, y), (255, 255, 255))
                im.putpixel((x, y), (0, 0, 0))
        if mode=='yellow':
            if r>maxthreshold and g>maxthreshold and b<101:
                #im.putpixel((x, y), (r, g, r))
                #im.putpixel((x, y), (255, 255, 255))
                im.putpixel((x, y), (COLOR, COLOR, COLOR))
            else:
                im.putpixel((x, y), (0, 0, 0))
        if mode=='red':
            if r>maxthreshold and g<119 and b<120:
                #im.putpixel((x, y), (r, r, r))
                im.putpixel((x, y), (COLOR, COLOR, COLOR))
            else:
                im.putpixel((x, y), (0, 0, 0))
        if mode=='void':
            if np.array([r,g,b]).mean()<110 and np.array([r,g,b]).std()<threshold :
                #im.putpixel((x, y), (255-r, 255-g, 255-b))
                im.putpixel((x, y), (COLOR, COLOR, COLOR))
            else:
                im.putpixel((x, y), (0, 0, 0))

def pic_convert(png_data,mode):
    fileobj = BytesIO()
    fileobj.write(png_data)
    image=Image.open(fileobj)
    image=image.resize((90, 36))
    enhance_color(image,mode)
    image= image.filter(ImageFilter.SMOOTH)
    #image,_,_ = image.split()
    image=image.convert('L')
    return image


def get_capcha_image_and_color(id):
    time_ms=time.time()*1000+100
    payload['_']=str('%d'%time_ms)
    payload['r']=str('%.16f'%random.random())
    payload['callback']='jQuery110205660534614759527_%d'%time_ms
    response = requests.get('https://zjfpcyweb.bjsat.gov.cn/WebQuery/yzmQuery',params=payload,headers=headers,verify=False)
    try:
        response=eval(re.findall('\((.*?)\)', response.text)[0])
    except Exception as e:
        print(e,response.text)
        return True,None
    imgdata=base64.b64decode(response['key1'])
    image_new=pic_convert(imgdata,clolr_map[response['key4']])

