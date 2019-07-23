import tensorflow as tf
import numpy as np
from text_rnn import TextRNN
import os
import time
import datetime
import data_helpers

################################################################
# print("trainsamples:",np.sum(train_label,axis=0))
# print("testsamples:",np.sum(test_label,axis=0))
# w2v_dim=300  # 一个单词向量的维度
# conv_filter=100  # num_filter
# batch_size =128
# batch_size_test=128
# drop_rate=0.8  # Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果
# train_size=train_data.shape[0]  # 序列的长度
# test_size=test_data.shape[0]
# epochs=50  # 总共把数据迭代多少轮，一轮代表把所有数据训练一次
################################################################

# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("hidden_dim", 64, "hidden dimension")
tf.flags.DEFINE_integer("num_layers", 2, "number of hidden layers")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)") # 每训练100次测试下
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)") # 保存一次模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")
# SVM parameters
tf.flags.DEFINE_float("gamma", 0.1, "svm parameter")
tf.flags.DEFINE_float("svm_c", 0.1, "svm parameter")

tf.flags.DEFINE_boolean("random", False, "Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static", False, "Keep the word embeddings static (default: False)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement") # 加上一个布尔类型的参数，要不要自动分配
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") # 加上一个布尔类型的参数，要不要打印日志

FLAGS = tf.flags.FLAGS

# Data Preparation
# =======================================================
train_data, train_label, test_data, test_label = data_helpers.data_processing(FLAGS.positive_data_file, FLAGS.negative_data_file)

print("training...")
# Training
# =======================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        # 卷积池化网络导入
        rnn = TextRNN(
            sequence_length=len(train_data[0]),
            num_classes=len(train_label[0]),
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            hidden_dim=FLAGS.hidden_dim,
            num_layers=FLAGS.num_layers)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)  # 计算梯度
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # 进行参数更新

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Summaries for loss and accuracy
        # 损失函数和准确率的参数保存
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        # 训练数据保存
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        # 测试数据保存
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        training_params_file = os.path.join(out_dir, 'training_params.pickle')
        params = {'num_labels': FLAGS.num_labels, 'max_document_length': len(train_data[0])}
        data_helpers.saveDict(params, training_params_file)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # define training function
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # define predict function
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy

        def test_step(x_batch):
            feed_dict = {rnn.input_x: x_batch,
                         rnn.dropout_keep_prob: 1.0}
            predictions = sess.run(rnn.y_pred_cls, feed_dict)
            return predictions

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(train_data, train_label)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                acc_val = dev_step(test_data, test_label, writer=dev_summary_writer)
                print("")
                acc_val_1 = 0
                if acc_val > acc_val_1:
                    acc_val_1 = acc_val

            # if current_step % FLAGS.checkpoint_every == 0:
                if acc_val > 0.78:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    break

        print("max_acc:{}".format(acc_val_1))