import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from tensorflow.contrib import rnn
from tensorflow.contrib import lookup

class matchJudge(object):
	def __init__(
		self, sequence_length, vocab_size, entity_vocab_size,
		entity_class_vocab_size, pos_vocab_size, embedding_size,
		pos_embedding_size, filter_sizes, num_filters, l2_reg_lambda, n_hidden_state):

		self.input_sentence = tf.placeholder(tf.int32, [None, sequence_length], name='input_sentence')
		self.input_result = tf.placeholder(tf.float32, [None, 2])
		self.entity1_class = tf.placeholder(tf.int32, [None, 1], name='entity1_class')
		self.entity2_class = tf.placeholder(tf.int32, [None, 1], name='entity2_class')
		self.entity1_id = tf.placeholder(tf.int32, [None, 1], name='entity1_id')
		self.entity2_id = tf.placeholder(tf.int32, [None, 1], name='entity2_id')
		self.entity1_pos = tf.placeholder(tf.int32, [None, 1], name='entity1_pos')
		self.entity2_pos = tf.placeholder(tf.int32, [None, 1], name='entity2_pos')
		self.relative_pos = tf.placeholder(tf.int32, [None, 1], name='pos_a_b')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.name_scope('embedding'):

			# initialize a word embedding matrix(word level)
			self.W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
				name="W")

			# transfer sentence into sentence embedding
			self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_sentence)

			# add a new dimension after the last dimension
			self.expanded_embedded_chars = tf.expand_dims(self.embedded_chars, -1)

			# initialize a position embedding matrix
			self.W_pos = tf.Variable(
				tf.random_uniform([pos_vocab_size, pos_embedding_size], -1.0, 1.0),
				name="W_pos")

			# initialize a entity embedding matrix(word level)
			self.W_entity = tf.Variable(
				tf.random_uniform([entity_vocab_size, embedding_size], -1.0, 1.0),
				name="W_entity")

			# initalize a entity type embedding matrix
			self.W_entity_class = tf.Variable(
				tf.random_uniform([entity_class_vocab_size, embedding_size], -1.0, 1.0),
				name="W_entity_class")

			# transfer position id into position embedding
			self.embedded_position = tf.nn.embedding_lookup(self.W_pos, self.relative_pos)
			self.embedded_entity1 = tf.nn.embedding_lookup(self.W_entity, self.entity1_id)
			self.embedded_entity2 = tf.nn.embedding_lookup(self.W_entity, self.entity2_id)
			self.embedded_entity1_class = tf.nn.embedding_lookup(self.W_entity_class, self.entity1_class)
			self.embedded_entity2_class = tf.nn.embedding_lookup(self.W_entity_class, self.entity2_class)

		with tf.name_scope('bi-lstm'):
			self.weight = tf.Variable(tf.random_normal([2 * n_hidden_state, 2]))
			self.bias =  tf.Variable(tf.random_normal([2]))
			self.embedded_chars = tf.unstack(self.embedded_chars, sequence_length, 1)
			lstm_fw_cell = rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0)
			lstm_bw_cell = rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0)
			outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_chars, dtype=tf.float32)
			self.hidden_output1 = None
			self.hidden_output2 = None
			for i in range(sequence_length):
				if self.hidden_output1 == None:
					self.hidden_output1 = tf.expand_dims(outputs[i], 0)
				else:
					self.hidden_output1 = tf.concat([self.hidden_output1, tf.expand_dims(outputs[i], 0)], 0)
				if self.hidden_output2 == None:
					self.hidden_output2 = tf.expand_dims(outputs[i], 0)
				else:
					self.hidden_output2 = tf.concat([self.hidden_output2, tf.expand_dims(outputs[i], 0)], 0)
			print(self.hidden_output1)
			self.entity1_pos = tf.squeeze(self.entity1_pos, axis=1)
			self.hidden_1 = tf.nn.embedding_lookup(self.hidden_output1, self.entity1_pos)
			self.entity2_pos = tf.squeeze(self.entity2_pos, axis=1)
			self.hidden_2 = tf.nn.embedding_lookup(self.hidden_output2, self.entity2_pos)
			self.hidden_1 = tf.transpose(self.hidden_1,[2,0,1])
			self.hidden_list_1 = list(map(tf.diag_part, [self.hidden_1[i] for i in range(2 * n_hidden_state)]))
			self.hidden_list_1 = [tf.expand_dims(_, 1) for _ in self.hidden_list_1]
			self.hidden_state1 = tf.concat(self.hidden_list_1, 1)
			self.hidden_2 = tf.transpose(self.hidden_2,[2,0,1])
			self.hidden_list_2 = list(map(tf.diag_part, [self.hidden_2[i] for i in range(2 * n_hidden_state)]))
			self.hidden_list_2 = [tf.expand_dims(_, 1) for _ in self.hidden_list_2]
			self.hidden_state2 = tf.concat(self.hidden_list_2, 1)			

		with tf.name_scope('attention'):
			self.entity_info = tf.concat(list(map(tf.squeeze, [
				self.embedded_entity1, self.embedded_entity2, 
				self.embedded_entity1_class, self.embedded_entity2_class])), 1)
			self.lexical_info = tf.concat(list(map(tf.squeeze, [self.hidden_state1, self.hidden_state2])), 1)
			self.pos_info = tf.squeeze(self.embedded_position, [1])
			entity_attention_matrix = tf.Variable(tf.random_uniform([embedding_size * 4, 1], -1.0, 1.0))
			lexical_attention_matrix = tf.Variable(tf.random_uniform([n_hidden_state * 4, 1], -1.0, 1.0))
			pos_attention_matrix = tf.Variable(tf.random_uniform([pos_embedding_size, 1], -1.0, 1.0))
			entity_attention = tf.matmul(self.entity_info, entity_attention_matrix)
			lexical_attention = tf.matmul(self.lexical_info, lexical_attention_matrix)
			pos_attention = tf.matmul(self.pos_info, pos_attention_matrix)
			self.entity_info_with_attention = entity_attention * self.entity_info
			self.lexical_info_with_attention = lexical_attention * self.lexical_info
			self.pos_info_with_attention = pos_attention * self.pos_info

		with tf.name_scope('concatenate'):
			self.vector = tf.concat([self.entity_info_with_attention, self.lexical_info_with_attention, self.pos_info_with_attention], 1)

		with tf.name_scope('dropout'):
			self.vector_drop = tf.nn.dropout(self.vector, self.dropout_keep_prob)

		with tf.name_scope('output'):
			W = tf.get_variable(
                'W',
                shape=[n_hidden_state * 4 + embedding_size * 4 + pos_embedding_size, 2],
                initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
			self.scores = tf.nn.xw_plus_b(self.vector_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_result)
			self.loss = tf.reduce_mean(losses)

		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_result, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		
		with tf.name_scope("confusion_matrix"):
			self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.input_result, 1), self.predictions)
