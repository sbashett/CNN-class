import tensorflow as tf
import numpy as np

class CNN(object):
	def __init__(self, num_classes, keep_prob  ):
		super(CNN, self).__init__()
		#self.NUM_SAMPLES = num_samples
		#self.WIDTH = width
		#self.HEIGHT = height
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob

	def conv_layer_relu(self,x, weights, biases, stride, name, relu = 'TRUE', padding = 'SAME'):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable("weights", weights, initializer = tf.truncated_normal_initializer())
			biases = tf.get_variable("biases", biases, initializer = tf.truncated_normal_initializer())

			conv = tf.nn.conv2d(x, weights, strides= stride, padding = padding, name = scope.name)

			if relu == 'TRUE':
				conv = tf.nn.relu(tf.add(conv, biases), name = scope.name + "_relu")

			return conv

	def maxpool(self,x, filter_size, stride,name):
		return tf.nn.max_pool(x, ksize = filter_size, strides = stride, padding = 'VALID', name = name)

	def fc_relu(self,x, weights, biases, name, relu = 'TRUE'):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable("weights", weights, initializer = tf.truncated_normal_initializer())
			biases = tf.get_variable("biases", biases, initializer = tf.truncated_normal_initializer())

			fc = tf.add(tf.matmul(x, weights ), biases, name = scope.name)

			if relu == 'TRUE':
				fc = tf.nn.relu(fc, name = scope.name + "_relu")
			return fc

	def dropout(self,x, name):
		return tf.nn.dropout(x, self.KEEP_PROB, name = name)

	def alex_net(self, x):
		
		#reshaping into 4d tensor		
		x = tf.reshape(x , [-1, 224,224,3])

		#conv1 layer with relu
		conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
		#maxpool_1
		pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
		#normalization layer after conv1
		norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

		#conv2 layer with relu
		conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
		#maxpool_2
		pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
		#normalization after conv2
		norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

		#conv3 layer with relu
		conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
		#conv4 layer with relu
		conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
		#conv5 layer with relu
		conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
		#maxpool_2 after conv5
		pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out

	def vgg_net(self,x):

		#reshaping into 4d tensor		
		x = tf.reshape(x , [-1, 224,224,3])

		#conv1_1 layer with relu
		conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
		#conv1_2 layer with relu
		conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
		#maxpool 1 
		pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
		#norm layer after pool1
		norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

		#conv2_1 layer with relu
		conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
		#conv2_2 layer with relu
		conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
		#maxpool 2 
		pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
		#norm layer after pool2
		norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

		#conv3_1 layer with relu
		conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
		#conv3_2 layer with relu
		conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
		#conv3_3 layer with relu
		conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
		#maxpool 3 
		pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
		#norm layer after pool3
		norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")

		#conv4_1 layer with relu
		conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
		#conv4_2 layer with relu
		conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
		#conv4_3 layer with relu
		conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
		#maxpool 4 
		pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
		#norm layer after pool4
		norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")

		#conv5_1 layer with relu
		conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
		#conv5_2 layer with relu
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
		#conv5_3 layer with relu
		conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
		#maxpool 5
		pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
		
		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out

	def resnet34(self,x):

		#reshaping into 4d tensor		
		x = tf.reshape(x , [-1, 224,224,3])

		conv1 = self.conv_layer_relu(x, [7,7,3,64], [64], [1,2,2,1], "res_conv1")
		pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "res_pool1")

		conv2_1 = self.conv_layer_relu(pool1, [3,3,64,64], [64], [1,1,1,1], "res_conv2_1")
		conv2_2 = self.conv_layer_relu(conv2_1, [3,3,64,64], [64], [1,1,1,1], "res_conv2_2", relu = 'FALSE')
		res_add1 = tf.add(pool1,conv2_2)
		res_add1 = tf.nn.relu(res1, name = 'res_add1')

		conv2_3 = self.conv_layer_relu(res_add1, [3,3,64,64], [64], [1,1,1,1], "res_conv2_3")
		conv2_4 = self.conv_layer_relu(conv2_3, [3,3,64,64], [64], [1,1,1,1], "res_conv2_4", relu = 'FALSE')
		res_add2 = tf.add(res_add1,conv2_4)
		res_add2 = tf.nn.relu(res_add2, name = 'res_add2')

		conv2_5 = self.conv_layer_relu(res_add2, [3,3,64,64], [64], [1,1,1,1], "res_conv2_5")
		conv2_6 = self.conv_layer_relu(conv2_5, [3,3,64,64], [64], [1,1,1,1], "res_conv2_6", relu = 'FALSE')
		res_add3 = tf.add(res_add2,conv2_6)
		res_add3 = tf.nn.relu(res_add3, name = 'res_add3')
		res_add3_conv = self.conv_layer_relu(res_add3, [1,1,64,128], [128], [1,2,2,1], "res_conv_res_add3", relu = 'FALSE', padding = 'VALID')

		conv3_1 = self.conv_layer_relu(res_add3, [3,3,64,128], [128], [1,2,2,1], "res_conv3_1")
		conv3_2 = self.conv_layer_relu(conv3_1, [3,3,128,128], [128], [1,1,1,1], "res_conv3_2", relu = 'FALSE')
		res_add4 = tf.add(res_add3_conv,conv3_2)
		res_add4 = tf.nn.relu(res_add4, name = 'res_add4')

		conv3_3 = self.conv_layer_relu(res_add4, [3,3,128,128], [128], [1,1,1,1], "res_conv3_3")
		conv3_4 = self.conv_layer_relu(conv3_3, [3,3,128,128], [128], [1,1,1,1], "res_conv3_4", relu = 'FALSE')
		res_add5 = tf.add(res_add4,conv3_4)
		res_add5 = tf.nn.relu(res_add5, name = 'res_add5')

		conv3_5 = self.conv_layer_relu(res_add5, [3,3,128,128], [128], [1,1,1,1], "res_conv3_5")
		conv3_6 = self.conv_layer_relu(conv3_5, [3,3,128,128], [128], [1,1,1,1], "res_conv3_6", relu = 'FALSE')
		res_add6 = tf.add(res_add5,conv3_6)
		res_add6 = tf.nn.relu(res_add6, name = 'res_add6')

		conv3_7 = self.conv_layer_relu(res_add6, [3,3,128,128], [128], [1,1,1,1], "res_conv3_7")
		conv3_8 = self.conv_layer_relu(conv3_7, [3,3,128,128], [128], [1,1,1,1], "res_conv3_8", relu = 'FALSE')
		res_add7 = tf.add(res_add6,conv3_8)
		res_add7 = tf.nn.relu(res_add7, name = 'res_add7')
		res_add7_conv = self.conv_layer_relu(res_add7, [1,1,128,256], [256], [1,2,2,1], "res_conv_res_add7", relu = 'FALSE', padding = 'VALID')

		conv4_1 = self.conv_layer_relu(res_add7, [3,3,128,256], [256], [1,2,2,1], "res_conv4_1")
		conv4_2 = self.conv_layer_relu(conv4_1, [3,3,256,256], [256], [1,1,1,1], "res_conv4_2", relu = 'FALSE')
		res_add8 = tf.add(res_add7_conv,conv4_2)
		res_add8 = tf.nn.relu(res_add8, name = 'res_add8')

		conv4_3 = self.conv_layer_relu(res_add8, [3,3,256,256], [256], [1,1,1,1], "res_conv4_3")
		conv4_4 = self.conv_layer_relu(conv4_3, [3,3,256,256], [256], [1,1,1,1], "res_conv4_4", relu = 'FALSE')
		res_add9 = tf.add(res_add8,conv4_4)
		res_add9 = tf.nn.relu(res_add9, name = 'res_add9')

		conv4_5 = self.conv_layer_relu(res_add9, [3,3,256,256], [256], [1,1,1,1], "res_conv4_5")
		conv4_6 = self.conv_layer_relu(conv4_5, [3,3,256,256], [256], [1,1,1,1], "res_conv4_6", relu = 'FALSE')
		res_add10 = tf.add(res_add9,conv4_6)
		res_add10 = tf.nn.relu(res_add10, name = 'res_add10')

		conv4_7 = self.conv_layer_relu(res_add10, [3,3,256,256], [256], [1,1,1,1], "res_conv4_7")
		conv4_8 = self.conv_layer_relu(conv4_7, [3,3,256,256], [256], [1,1,1,1], "res_conv4_8", relu = 'FALSE')
		res_add11 = tf.add(res_add10,conv4_8)
		res_add11 = tf.nn.relu(res_add11, name = 'res_add11')

		conv4_9 = self.conv_layer_relu(res_add11, [3,3,256,256], [256], [1,1,1,1], "res_conv4_9")
		conv4_10 = self.conv_layer_relu(conv4_9, [3,3,256,256], [256], [1,1,1,1], "res_conv4_10", relu = 'FALSE')
		res_add12 = tf.add(res_add11,conv4_10)
		res_add12 = tf.nn.relu(res_add12, name = 'res_add12')

		conv4_11 = self.conv_layer_relu(res_add12, [3,3,256,256], [256], [1,1,1,1], "res_conv4_11")
		conv4_12 = self.conv_layer_relu(conv4_11, [3,3,256,256], [256], [1,1,1,1], "res_conv4_12", relu = 'FALSE')
		res_add13 = tf.add(res_add12,conv4_12)
		res_add13 = tf.nn.relu(res_add13, name = 'res_add13')
		res_add13_conv = self.conv_layer_relu(res_add13, [1,1,256,512], [512], [1,2,2,1], "res_conv_res_add13", relu = 'FALSE', padding = 'VALID')

		conv5_1 = self.conv_layer_relu(res_add13, [3,3,256,512], [512], [1,2,2,1], "res_conv5_1")
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "res_conv5_2", relu = 'FALSE')
		res_add14 = tf.add(res_add13_conv,conv5_2)
		res_add14 = tf.nn.relu(res_add14, name = 'res_add14')

		conv5_3 = self.conv_layer_relu(res_add14, [3,3,512,512], [512], [1,1,1,1], "res_conv5_3")
		conv5_4 = self.conv_layer_relu(conv5_3, [3,3,512,512], [512], [1,1,1,1], "res_conv5_4", relu = 'FALSE')
		res_add15 = tf.add(res_add14,conv5_4)
		res_add15 = tf.nn.relu(res_add15, name = 'res_add15')

		conv5_5 = self.conv_layer_relu(res_add15, [3,3,512,512], [512], [1,1,1,1], "res_conv5_5")
		conv5_6 = self.conv_layer_relu(conv5_5, [3,3,512,512], [512], [1,1,1,1], "res_conv5_6", relu = 'FALSE')
		res_add16 = tf.add(res_add15,conv5_6)
		res_add16 = tf.nn.relu(res_add16, name = 'res_add16')	#need to check if relu required here

		pool2 = tf.nn.pool(res_add16, window_shape = [1,7,7,1], pooling_type = 'AVG' , padding = 'VALID')

		#reshaping for fc layers
		x2 = tf.reshape(pool2, [-1, 7*7*512])

		out = self.fc_relu(x2, [7*7*512, self.NUM_CLASSES],[self.NUM_CLASSES], "res_out", relu = 'FALSE')

		return out


