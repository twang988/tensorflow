
import tensorflow as tf

from cfg import MAX_CAPTCHA, CHAR_SET_LEN, tb_log_path, save_model,model_path
from cnn_sys import crack_captcha_cnn, Y, keep_prob, X
from data_iter import get_next_batch
from predict import test_hack_captcha_training_data
import time,os
def train_crack_captcha_cnn():
    """
    训练模型
    :return:
    """
    global_step = tf.Variable(0, trainable=False)
    output = crack_captcha_cnn()
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)  # shape:batch_size,4,nb_cls
    max_idx_l = tf.argmax(label, 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('my_monitor'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label))#最大概率分类
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))#二进制分类
    tf.summary.scalar('my_loss', loss)


    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss,global_step=global_step)

    with tf.name_scope('my_monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('my_accuracy', accuracy)


    with tf.device('/gpu:0'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=2)  # 将训练过程进行保存
    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    except Exception as e:
        print (e)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tb_log_path, sess.graph)

    step = 0
    time_ep_start=time.time()
    max_acc_on_tax_pic=0
    max_acc_step=0
    while True:
        batch_x, batch_y = get_next_batch(64)  # 64
        _, loss_ ,step= sess.run([optimizer, loss,global_step], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})

        if step%10==0:
            print(step, 'loss: ', loss_)


        # 每100步保存一次实验结果
        if step % 100 == 0:
            saver.save(sess, save_model, global_step=step)
        else:
            continue
        # 生成测试集计算精度
        ep_time=time.time()-time_ep_start
        time_ep_start=time.time()
        acc_train_pic=test_hack_captcha_training_data(sess,output)
        print('EP spend:%0.2fs\n'%ep_time,
            'acc_on_train_pic: %0.2f%%'%(acc_train_pic*100)
            )

        # 终止条件
        if acc_train_pic>0.98:
            break


if __name__ == '__main__':
    train_crack_captcha_cnn()
    #get_next_batch(100)
    print('end')
    pass
