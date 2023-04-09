
import tensorflow as tf

#===================
#定义参数、权重和偏置
#===================
# Parameters参数
learning_rate = 0.0001
training_epochs = 5
batch_size = 50000
display_step = 1
# Network Parameters网络结构
n_input = 43
n_hidden_1 = 60  # 1st layer number of features
n_hidden_2 = 60  # 2nd layer number of features
n_classes = 2#二分类
# tf Graph input 输入
x = tf.placeholder("float", [None, n_input])#占位，输入维度和数据类型
y = tf.placeholder("float", [None, n_classes])#占位，输出维度和数据类型
# Store layers weight & bias 权重和偏置
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


#====================
#前向传播算法，使用ReLU激活函数，并进行Dropout降采样
#====================
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])#tf.matmul矩阵乘法，然后各个维度加上偏置
    layer_1 = tf.nn.relu(layer_1)#第一个隐层的输出
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)#第二个隐层的输出
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)
    return out_layer


#=======================
#模型训练和存储
#=======================
def multilayer_perceptron_train_save(x, weights, biases,x_train, y_train):
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    saver = tf.train.Saver()# Initiate tf saver
    model_path =r'E:\all_models\model.ckpt'
    sess = tf.InteractiveSession()# Starting Session
    tf.global_variables_initializer().run()# Initializing the variables
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={x: x_train, y : y_train})
        cc = sess.run(cost, feed_dict={x: x_train, y : y_train})
        if epoch % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))# Test model
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_percentage = sess.run(accuracy, feed_dict={x: x_train, y : y_train})
            print("Training Step: ", "%04d" % (epoch), 'cost=', "{:.9f}".format(cc), "Accuracy: ", accuracy_percentage)
            xpoints.append(epoch)
            ypoints.append(accuracy_percentage * 100)
    print("Optimization Finished!")
    print(xpoints, ypoints)
    plt.plot(xpoints, ypoints)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Percentage')
    plt.title("Model Accuracy vs No of Epochs")
    plt.legend()
    plt.show()
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    return save_path
