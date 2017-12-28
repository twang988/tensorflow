# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import numpy as np
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None



DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]

if wheezy_captcha:
    __all__ = ['ImageCaptcha', 'WheezyCaptcha']
else:
    __all__ = ['ImageCaptcha']


table  =  []
for  i  in  range( 256 ):
    table.append(255)
    #table.append( i * 1.97 )


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destionation.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class WheezyCaptcha(_Captcha):
    """Create an image CAPTCHA with wheezy.captcha."""
    def __init__(self, width=200, height=75, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                wheezy_captcha.curve(),
                wheezy_captcha.noise(),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)



class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=90, height=36, fonts=None,fonts2=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts #or DEFAULT_FONTS
        self._fonts2=fonts2#for chinese
        self._font_sizes = font_sizes or (32,35,42)
        self._truefonts = []
        self._truefonts2 = []
    @property
    def truefonts(self):
        #if self._truefonts:
        #    return self._truefonts
        self._truefonts = tuple([
            truetype(n, random.randint(45,55))
            #truetype(n, s)
            for n in self._fonts
            #for s in self._font_sizes
        ])
        return self._truefonts
    @property
    def truefonts2(self):

        self._truefonts2 = tuple([
            truetype(n, random.randint(42,47))
            for n in self._fonts2
        ])
        return self._truefonts2
    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=1, number=10):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)
        def is_chinese(uchar):
            if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
            else:
                return False
        def _draw_character(c,font):
            #if is_chinese(c):
                #font = random.choice(self.truefonts(index=1))
            #else:
                #font = random.choice(self.truefonts(index=0))
            #font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            #dx = random.randint(0, 4)
            #dy = random.randint(0, 6)
            #im = Image.new('RGBA', (w + dx, h + dy))
            im = Image.new('RGBA', (w, h))
            #Draw(im).text((dx, dy), c, font=font, fill=color)
            Draw(im).text((0, 0), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())

            im = im.rotate(random.uniform(-50, 50), Image.BILINEAR,expand=1)
            #return im
            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))

            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []

        for c in chars:
            if not c.encode('utf-8').isalnum():#中文
                font = random.choice(self.truefonts2)
            else:
                font = random.choice(self.truefonts)
            images.append(_draw_character(c,font))


        text_width = sum([im.size[0] for im in images])
        width = max(text_width, self._width)
        #image = image.resize((width, self._height))
        average = int(text_width / len(chars))
        rand = int(0.1 * average)
        #offset = int(average * 0.1)
        def left_w(last_index):
            _left_w=0
            index_=0
            if last_index==-1:
                return text_width
            for im in images:
                if index_<=last_index:
                    index_+=1
                    continue
                _left_w+=im.size[0]
                index_+=1

            return _left_w

        offset=0
        offset=random.randint(offset,width-text_width)
        #print('chars',chars)
        index_=0
        for im in images:

            w, h = im.size
            rand_y_off=0
            if self._height>h: rand_y_off=random.randint(-7,self._height-h)
            image.paste(im, (offset,rand_y_off ),mask=Image.merge("L", (im.split()[3],)))
            off=self._width-(offset+w)-left_w(index_)
            if off<0: off=0
            offset = offset+w+random.randint(-int(w/2),off)
            index_+=1

        #if image.size[0] > self._width :
            #image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        bg=random.randint(0, 70)
        background = (bg,bg,bg)#random_color(238, 255)
        #color = (random.randint(250, 255),
        #random.randint(250, 255),
        #random.randint(250, 255))#random_color(0, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, (255,255,255), background)
        self.create_noise_dots(im, (255,255,255),width=2,number=50)
        self.create_noise_dots(im, (0,0,0),width=2,number=50)
        self.create_noise_curve(im, (0,0,0))
        self.create_noise_curve(im, (0,0,0))
        #im = im.filter(ImageFilter.SMOOTH)#remove smooth
        return im


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)
