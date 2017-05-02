import tensorflow as tf
from os import listdir
from os.path import isfile, join

graph_file_name = '/root/projects/dogvscat/model/classify_image_graph_def.pb'
input_dir = '/root/projects/dogvscat/test'
prediction_list = []
labels=['cat', 'dog']

image_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

def predict_on_image(image, labels):

    # Unpersists graph from file
    with tf.gfile.FastGFile(graph_file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image, 'rb').read()

        try:
            predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
            prediction = predictions[0]
        except:
            print("Error making prediction.")
            sys.exit()

        # Return the label of the top classification.
        prediction = prediction.tolist()
        max_value = max(prediction)
        max_index = prediction.index(max_value)
        predicted_label = labels[max_index]
        
        return prediction

for i in range(0,1000):
    prediction_list.append(predict_on_image(image_files[i], labels))

prediction_list
