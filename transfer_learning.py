"""
我们把网上训练好的网络下载下来，替换成我们自己的输出层。训练的时候只训练这一层，之前的部分视为一个“bottleneck”， 
不参与训练，所以对于每张图片bottleneck的输出都是恒定不变的，我们吧图片对应bottleneck输出保存成文件。
bottleneck目录结构和image目录结构相同。/mulu/aaa.jpg   --->/目录/aaa.jpg.txt
"""
#这是Inception v3的相关信息
BOTTLENECK_TENSOR_SIZE =2048
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_PATH = "/home/Liuch/inception/inception_dec/tensorflow_inception_graph.pb"
CACHE_DIR = '/home/Liuch/inception/cache'
INPUT_DIR = '/home/Liuch/inception/flower_photos'

LEARNING_RATE = 0.01
EPOCHES = 4000
BATCH_SIZE = 100

#storage struct
#{'training' : array([{'label_1': fn_1}, {'label_0' : fn_2}...]),
#  'testing':array([{'label_1': fn_1}, {'label_0' : fn_2}...])，
# 'validation': array([{'label_1': fn_1}, {'label_0' : fn_2}...])}

#记录label array每个维度表示的具体类别，预测时方便
#{'sunflowers': 1, 'dandelion': 3, 'tulips': 2, 'daisy': 0, 'roses': 4}


def create_sampleSpace(training_percentage, test_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DIR)]
    sub_dirs = sub_dirs[1:]
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    training_list = []
    testing_list = []
    validation_list = []
    label_pos={}
    for i in range(len(sub_dirs)):
        sub_dir = sub_dirs[i]
        # dir name is label name
        label_name = os.path.basename(sub_dir)
        label_pos[label_name] = i
        
        filelists = []
        for ext in extensions:
            file_glob =  os.path.join(sub_dir, '*.'+ext)
            filelists.extend(glob.glob(file_glob))

        filelists = np.asarray(filelists)
        permutation = np.random.permutation(filelists.shape[0])

        training_list.extend(
            [{label_name: fn} for fn in filelists[permutation[: int(training_percentage * filelists.shape[0])]]])
        testing_list.extend(
            [{label_name: fn} for fn in filelists[permutation[int(training_percentage * filelists.shape[0]): \
            int(training_percentage * filelists.shape[0]) + int(test_percentage * filelists.shape[0])]]])
        validation_list.extend(
            [{label_name: fn} for fn in filelists[permutation[int(training_percentage * filelists.shape[0]) \
                                                                            + int(test_percentage * filelists.shape[0]):]]])

    result = {'training': np.asarray(training_list), 'testing': np.asarray(testing_list), 'validation': np.asarray(validation_list)}
    np.random.shuffle(result['training'])
    np.random.shuffle(result['testing'])
    np.random.shuffle(result['validation'])
    return result, label_pos
def run_bottleneck_on_image(sess, image_data, input_tensor, bottleneck_tensor):
    bottleneck = sess.run(bottleneck_tensor, {input_tensor: image_data})
    #输出是四维数组，要展成一维数组的形式
    bottleneck = np.squeeze(bottleneck.reshape(1,-1))
    return bottleneck


def get_bottleneck(sess, 
                   image_path,
                   input_tensor, bottleneck_tensor):
    path_split = image_path.split('/')
    label_name = path_split[-2]
    image_name = path_split[-1]
    cache_dir = os.path.join(CACHE_DIR, label_name)
    #如果连目录都不存在先创建目录
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    botteneck_path = os.path.join(cache_dir, image_name +'.txt') 
    #文件不存在，运行出bottleneck，写入文件并返回
    if not os.path.exists(botteneck_path):
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck = run_bottleneck_on_image(sess, image_data, input_tensor, bottleneck_tensor)
        
        bottleneck_str = ','.join([str(x) for x in bottleneck])
        with open(botteneck_path, 'w') as f:
            f.write(bottleneck_str)
    #bottleneck文件已经存在，直接读出返回
    else:
        with open(botteneck_path, 'r') as f:
            bottleneck_str = f.read()
        bottleneck = np.asarray([float(x) for x in bottleneck_str.split(',')])
    
    return bottleneck

start = 0
def next_batch(batch_size, sess, input_tensor, bottleneck_tensor):
    global start
    global image_lists
    
    start += batch_size
    bottlenecks = []
    labels = []
    if start > image_lists['training'].shape[0]:
        start = 0
        np.random.shuffle(image_lists['training'])
    for d in image_lists['training'][start : start+batch_size]:
        label_name = list(d.keys())[0]
        image_path = list(d.values())[0]
        label= np.zeros((len(label_pos)), dtype=np.float32)
        label[label_pos[label_name]] = 1.0
        labels.append(label)
        bottleneck = get_bottleneck(sess, image_path, input_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
    return np.asarray(bottlenecks).reshape(len(bottlenecks), -1), np.asarray(labels).reshape(len(labels), -1)


def get_validation_test_bottlenecks(sess, cataglory, input_tensor, bottleneck_tensor):
    bottlenecks = []
    labels = []
    test_size = image_lists[cataglory].shape[0]
    for d in image_lists[cataglory]:
        label_name = list(d.keys())[0]
        image_path = list(d.values())[0]
        label= np.zeros((len(label_pos)), dtype=np.float32)
        label[label_pos[label_name]] = 1.0
        labels.append(label)
        
        bottleneck = get_bottleneck(sess, image_path, input_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
        
    return np.asarray(bottlenecks).reshape(test_size, -1), np.asarray(labels).reshape(test_size, -1)


with gfile.FastGFile(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    bottleneck_tensor, input_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME])

bottleneck_output =  tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], "bottleneck_output")
Y = tf.placeholder(tf.float32, [None, len(label_pos)], "Y")


with tf.name_scope('output_layer'):
    w = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, len(label_pos)], stddev=0.001))
    b = tf.Variable(tf.zeros([len(label_pos)]))
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(0.0003)(w))
    logits = tf.matmul(bottleneck_output, w) + b
    output = tf.nn.softmax(logits)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
cost += tf.add_n(tf.get_collection("loss"))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE, 
                                           global_step, 
                                           image_lists['training'].shape[0]/BATCH_SIZE, 
                                           0.99)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost, global_step=global_step)

with tf.name_scope('evaluztion'):
    prediction = tf.equal(tf.argmax(Y,1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))



with tf.device('/gpu:0'):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(EPOCHES):
            bottlenecks, labels = next_batch(BATCH_SIZE, sess, input_tensor, bottleneck_tensor)
            c, _ = sess.run([cost, optimizer], {bottleneck_output: bottlenecks, Y: labels})
            if epoch %100 == 0:
                print('%d epoch, train cost %g' %(epoch, c/BATCH_SIZE))
                bottlenecks, labels =  get_validation_test_bottlenecks(sess, 'validation', input_tensor, bottleneck_tensor)
                acc = accuracy.eval({bottleneck_output: bottlenecks, Y: labels})
                print("%d epoch, accuracy on validation : %g" %(epoch, acc))
                print('----------------------------------------------')
                if acc > 0.94:
                    break
        bottlenecks, labels =  get_validation_test_bottlenecks(sess, 'testing', input_tensor, bottleneck_tensor)
        print("%d epoch, accuracy on test : %g" %(epoch, accuracy.eval({bottleneck_output: bottlenecks, Y: labels})))
