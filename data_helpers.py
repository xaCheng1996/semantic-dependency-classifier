import os
import numpy as np

def load_data(data_file_name):
	with open(data_file_name, 'r', ) as input_file:
		input_sentences = []
		entity1_name_lot = ''
		entity2_name_lot = ''
		entity1_class_lot = ''
		entity2_class_lot = ''
		# entity1_between_lot = ''
		# entity2_between_lot = ''
		# entity3_between_lot = ''
		entity1_pos_lot = []
		entity2_pos_lot = []
		rel_pos = ''
		input_result = None
		lines = input_file.readlines()
		index = 0
		for line in lines:
			sentence, entity1_name, entity2_name, entity1_class, entity2_class, entity1_pos, entity2_pos, pos, result = line.strip('\n').split('/')
			token_sentence = ''
			for i in sentence:
				token_sentence += i
				token_sentence += ' '
			# token_sentence = sentence
			input_sentences.append(token_sentence)
			entity1_name_lot += entity1_name + ' '
			entity2_name_lot += entity2_name + ' '
			entity1_class_lot += entity1_class + ' '
			entity2_class_lot += entity2_class + ' '
			# entity_between_list = entity_between.strip('[]').split(',')
			# entity1_between_lot += entity_between_list[0] + ' '
			# entity2_between_lot += entity_between_list[1] + ' '
			# entity3_between_lot += entity_between_list[2] + ' '
			entity1_pos_lot.append(int(entity1_pos))
			entity2_pos_lot.append(int(entity2_pos))
			rel_pos += pos + ' '
			if index == 0:
				if result == 'True':
					input_result = np.array([[1,0]])
				else:
					input_result = np.array([[0,1]])
				index += 1
			else:
				if result == 'True':
					input_result = np.concatenate((input_result, np.array([[1,0]])))
				else:
					input_result = np.concatenate((input_result, np.array([[0,1]])))
	return [input_sentences, entity1_name_lot, entity2_name_lot, entity1_class_lot, entity2_class_lot, entity1_pos_lot, entity2_pos_lot, rel_pos, input_result]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = -(-data_size // batch_size)
	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indicies = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indicies]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index : end_index]
