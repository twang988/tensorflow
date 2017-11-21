"""
网络结构
"""
import tensorflow as tf
import math
from cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout




def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """
    定义CNN
    cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
    np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
    """

    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #机器好的可以改为
    #L1_NEU_NUM=256 
    #L2_NEU_NUM=512
    #L3_NEU_NUM=1024
    #L4_NEU_NUM=2048
    #CONV_CORE_SIZE=5
    #NEU_LAYER_NUM=4
    #FULL_LAYER_FEATURE_NUM=8192
    
    
    
    L1_NEU_NUM=64
    L2_NEU_NUM=128
    L3_NEU_NUM=256
    L4_NEU_NUM=512
    CONV_CORE_SIZE=5
    NEU_LAYER_NUM=4
    FULL_LAYER_FEATURE_NUM=1024
    w_c1 = tf.Variable(w_alpha * tf.random_normal([CONV_CORE_SIZE, CONV_CORE_SIZE, 1, L1_NEU_NUM]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([L1_NEU_NUM]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([CONV_CORE_SIZE, CONV_CORE_SIZE, L1_NEU_NUM, L2_NEU_NUM]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([L2_NEU_NUM]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([CONV_CORE_SIZE, CONV_CORE_SIZE, L2_NEU_NUM, L3_NEU_NUM]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([L3_NEU_NUM]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_c4 = tf.Variable(w_alpha * tf.random_normal([CONV_CORE_SIZE, CONV_CORE_SIZE, L3_NEU_NUM, L4_NEU_NUM]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([L4_NEU_NUM]))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # Fully connected layer

    r=int(math.ceil(IMAGE_HEIGHT/(2**NEU_LAYER_NUM))*math.ceil(IMAGE_WIDTH/(2**NEU_LAYER_NUM))*L4_NEU_NUM)
    w_d = tf.Variable(w_alpha * tf.random_normal([r, FULL_LAYER_FEATURE_NUM]))
    b_d = tf.Variable(b_alpha * tf.random_normal([FULL_LAYER_FEATURE_NUM]))
    dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([FULL_LAYER_FEATURE_NUM, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)  # 36*4
    return out

