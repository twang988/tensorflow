"""
参考文章：http://blog.topspeedsnail.com/archives/10858
"""
import tensorflow as tf

from cfg import MAX_CAPTCHA, CHAR_SET_LEN, tb_log_path, save_model,model_path,IMAGE_HEIGHT, IMAGE_WIDTH
from cnn_sys import Y, keep_prob, X,training_flag,crack_captcha_cnn
from data_iter import get_next_batch,get_next_batch_from_web
from predict_tax_pic import test_hack_captcha
from predict import test_hack_captcha_training_data
import time,os
def train_crack_captcha_cnn():
    """
    训练模型
    :return:
    """
    global_step = tf.Variable(0, trainable=False)

    output = crack_captcha_cnn().model


    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    label=tf.placeholder(tf.float32, shape=[None, MAX_CAPTCHA*CHAR_SET_LEN])

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))

    with tf.name_scope('my_monitor'):
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))#多分类问题
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label))#二分类问题
    tf.summary.scalar('my_loss', loss)
    # 最后一层用来分类的softmax和sigmoid有什么不同？

    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss,global_step=global_step)


    with tf.name_scope('my_monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('my_accuracy', accuracy)


    with tf.device('/gpu:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement=False
        sess = tf.Session(config=config)

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
    max_acc_on_train_pic=0
    max_acc_step=0
    while True:
        batch_x, batch_y = get_next_batch_from_web(64)  # 64
        _, loss_ ,step= sess.run([optimizer,loss,global_step], feed_dict={X: batch_x, label: batch_y, training_flag:True})

        if step%10==0:
            print(step, 'loss: ', loss_)
        # 每200步保存一次实验结果
        if step % 100 == 0:
            saver.save(sess, save_model, global_step=step)
        else:
            continue
        # 在测试集上计算精度
        ep_time=time.time()-time_ep_start
        time_ep_start=time.time()
        acc_train_pic=test_hack_captcha_training_data(sess,output)
        #acc_tax_pic=test_hack_captcha(sess,global_step,output)

        if max_acc_on_train_pic<acc_train_pic:
            max_acc_on_train_pic=acc_train_pic
            max_acc_step=step
        print('EP spend:%0.2fs\n'%ep_time,
            #'acc_on_tax_pic: %0.2f%%\n'%(acc_tax_pic*100),
            'acc_on_train_pic: %0.2f%%\n'%(acc_train_pic*100),
            'max_acc_on_train_pic:%0.2f%%,on step:%d'%(max_acc_on_train_pic*100,max_acc_step)
            )

        '''
        if acc_tax_pic>max_acc_on_tax_pic:
            if max_acc_on_tax_pic>=0.91:
                cmd='cp ./model/*%d*  ./model_max_acc_on_tax/'%step
                os.popen(cmd)
            max_acc_step=step
            max_acc_on_tax_pic=acc_tax_pic

        print('max_acc_on_tax_pic: %0.2f%% on step:%d'%(max_acc_on_tax_pic*100,max_acc_step))
        '''
        # 终止条件
        if acc_train_pic==1.0:
            pass
            #break

        # 启用监控 tensor board
        #summary = sess.run(merged, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        #writer.add_summary(summary, step)


if __name__ == '__main__':
    train_crack_captcha_cnn()
    #get_next_batch(100)
    print('end')
    pass
