import tensorflow as tf

print(tf.__version__)
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    with tf.io.gfile.GFile("./sModel/trained.pb", "rb") as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    # print(graph_def)
    count = 0
    for op in sess.graph.get_operations():
        print(op.name)
        count +=1
        if count >= 5000 :
            print(op.name)
    morph = tf.placeholder(tf.int32, [None, None], name = "morph")
    ne_dict = tf.placeholder(tf.float32, [None, None, int(6 / 2)],name = "ne_dict")
    character = tf.placeholder(tf.int32, [None, None, None],name = "character")
    dropout_rate = tf.placeholder(tf.float32,name = "dropout_rate")
    sequence = tf.placeholder(tf.int32, [None],name = "sequence")
    character_len = tf.placeholder(tf.int32,[None, None], name = "character_len")
    label = tf.placeholder(tf.int32, [None, None],name = "label")
    global_step = tf.Variable(0, trainable=False,name = "global_step")
    seq_len = tf.placeholder(tf.int32, [None],name = "seq_len")
    feed_dict = {"morph": morph,
                 "ne_dict": ne_dict,
                 "character": character,
                 "sequence": seq_len,
                 "character_len": character_len,
                 "label": label,
                 "dropout_rate": dropout_rate
                 }

    input_tensor = feed_dict
    output_tensor = tf.placeholder(tf.int32, [None, None])
    # [morph],[ne_dict],[character],[sequence],[character_len],[label],[dropout_rate]
    tflite_model = tf.compat.v1.lite.toco_convert(sess.graph_def,[morph],[ne_dict],[character],[sequence],[character_len],[label],[global_step], [output_tensor])
    open("./sModel/tfliteModel", "wb").write(tflite_model)