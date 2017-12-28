from os.path import join
f=open('./chinese.txt','r')
CH_CHAR=[]
lines=f.read()
f.close()
CH_CHAR=eval(lines)

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

#gen_char_set = number+ ALPHABET +CH_CHAR + ['_'] # 用于生成验证码的数据集
#gen_char_set=number+ALPHABET+['_']
gen_char_set=number#+ALPHABET
# 有先后的顺序的

# 图像大小
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 120


MAX_CAPTCHA = 8  # 验证码位数
print("验证码文本最长字符数", MAX_CAPTCHA)  # 如果验证码长度小于MAX_CAPTCHA，用'_'补齐

CHAR_SET_LEN = len(gen_char_set)

print('CHAR_SET_LEN:', CHAR_SET_LEN)

home_root = './'  # 在不同操作系统下面Home目录不一样
model_path = join(home_root, 'model')
model_tag = 'crack_capcha.model'
save_model = join(model_path, model_tag)
train_pic_path='./train_pic/'

print('model_path:', save_model)

# 输出日志 tensorboard监控的内容
tb_log_path = '/tmp/mnist_logs'
