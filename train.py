import tensorflow as tf
import numpy as np
import os
import time
import data_helpers
from model import matchJudge
from tensorflow.contrib import learn
import datetime


tf.flags.DEFINE_float('train_sample_percentage', 0.8, 'train sample percentage')
tf.flags.DEFINE_float('validate_sample_percentage', 0.1, 'test sample percentage')
tf.flags.DEFINE_string('input_file_name','data/balanced_training_data.txt' ,'input_file_name')
tf.flags.DEFINE_string('test_file_name', 'data/full_generated_test_data.txt', 'test_file_name')
tf.flags.DEFINE_integer('embedding_dim', 300, 'dimension for word embedding')
tf.flags.DEFINE_boolean('use_trained_embedding', False, 'use pre-trained embedding')
tf.flags.DEFINE_integer('pos_emnedding_dim', 100, 'dimension for pos embedding')
tf.flags.DEFINE_string('embedding_file', '', 'embedding file path')
tf.flags.DEFINE_integer('num_filters', 100, 'number per filter size')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'dropout keep probability')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'l2 regularization lmabda')
tf.flags.DEFINE_integer('n_hidden_state', 128, 'n hidden state')

tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 20, 'num epochs')
tf.flags.DEFINE_integer('evaluate_every', 25, 'evaluate model on dev set every')
tf.flags.DEFINE_integer('checkpoint_every', 25, 'save model every')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'number of checkpoints')

tf.flags.DEFINE_boolean('allow_soft_placement', True, 'allow soft replacement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'log device placement')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
	print('{}={}'.format(attr.upper(), value))
print('')

print('Loading data...')
input_sentences, entity1_names, entity2_names, entity1_classes, entity2_classes, entity1_pos_lot, entity2_pos_lot, rel_pos, input_result = data_helpers.load_data(FLAGS.input_file_name)
input_sentences_test, entity1_names_test, entity2_names_test, entity1_classes_test, entity2_classes_test, entity1_pos_lot_test, entity2_pos_lot_test, rel_pos_test, input_result_test = data_helpers.load_data(FLAGS.test_file_name)
max_document_length = max([len(x.split()) for x in input_sentences])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
sentence_matrix = np.array(list(vocab_processor.fit_transform(input_sentences)))
sentence_matrix_test = np.array(list(vocab_processor.fit_transform(input_sentences_test)))

entity_vocab_processor = learn.preprocessing.VocabularyProcessor(1)
entity1_matrix = np.array(list(entity_vocab_processor.fit_transform(entity1_names)))[ : , 0 : 1]
entity2_matrix = np.array(list(entity_vocab_processor.fit_transform(entity2_names)))[ : , 0 : 1]
entity1_matrix_test = np.array(list(entity_vocab_processor.fit_transform(entity1_names_test)))[ : , 0 : 1]
entity2_matrix_test = np.array(list(entity_vocab_processor.fit_transform(entity2_names_test)))[ : , 0 : 1]

entity_class_vocab_processor = learn.preprocessing.VocabularyProcessor(1)
entity1_class_matrix = np.array(list(entity_class_vocab_processor.fit_transform(entity1_classes)))
entity2_class_matrix = np.array(list(entity_class_vocab_processor.fit_transform(entity2_classes)))
entity1_class_matrix_test = np.array(list(entity_class_vocab_processor.fit_transform(entity1_classes_test)))
entity2_class_matrix_test = np.array(list(entity_class_vocab_processor.fit_transform(entity2_classes_test)))

pos_vocab_processor = learn.preprocessing.VocabularyProcessor(1)
pos_matrix = np.asarray(list(pos_vocab_processor.fit_transform(rel_pos)))
pos_matrix_test = np.asarray(list(pos_vocab_processor.fit_transform(rel_pos_test)))

np.random.seed()
shuffle_indicies = np.random.permutation(np.arange(len(sentence_matrix)))
shuffled_sentences = sentence_matrix# [shuffle_indicies]
shuffled_entity1 = entity1_matrix# [shuffle_indicies]
shuffled_entity2 = entity2_matrix# [shuffle_indicies]
shuffled_entity1_class = entity1_class_matrix# [shuffle_indicies]
shuffled_entity2_class = entity2_class_matrix# [shuffle_indicies]
shuffled_entity1_pos = entity1_pos_lot# [shuffle_indicies]
shuffled_entity2_pos = entity2_pos_lot# [shuffle_indicies]
shuffled_pos = pos_matrix# [shuffle_indicies]
shuffled_result = input_result# [shuffle_indicies]

train_sample_index = int(len(shuffle_indicies)) - 1500
validate_sample_index = train_sample_index + 1500
sen_train, e1_train, e2_train, e1_class_train, e2_class_train, e1_pos_train, e2_pos_train, pos_train, res_train  = shuffled_sentences[ : train_sample_index], shuffled_entity1[ : train_sample_index], shuffled_entity2[ : train_sample_index], shuffled_entity1_class[ : train_sample_index], shuffled_entity2_class[ : train_sample_index], shuffled_entity1_pos[ : train_sample_index], shuffled_entity2_pos[ : train_sample_index], shuffled_pos[ : train_sample_index], shuffled_result[ : train_sample_index]
sen_val, e1_val, e2_val, e1_class_val, e2_class_val, e1_pos_val, e2_pos_val, pos_val, res_val = shuffled_sentences[train_sample_index : validate_sample_index], shuffled_entity1[train_sample_index : validate_sample_index], shuffled_entity2[train_sample_index : validate_sample_index], shuffled_entity1_class[train_sample_index : validate_sample_index], shuffled_entity2_class[train_sample_index : validate_sample_index],shuffled_entity1_pos[train_sample_index : validate_sample_index], shuffled_entity2_pos[train_sample_index : validate_sample_index], shuffled_pos[train_sample_index : validate_sample_index], shuffled_result[train_sample_index : validate_sample_index]

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		judge = matchJudge(
			sequence_length=max_document_length,
			vocab_size=len(vocab_processor.vocabulary_),
			entity_vocab_size=len(entity_vocab_processor.vocabulary_),
			entity_class_vocab_size=len(entity_class_vocab_processor.vocabulary_),
			pos_vocab_size=len(pos_vocab_processor.vocabulary_),
			embedding_size=FLAGS.embedding_dim,
			pos_embedding_size=FLAGS.pos_emnedding_dim,
			filter_sizes=[4,6,8],
			num_filters=FLAGS.num_filters,
			l2_reg_lambda=FLAGS.l2_reg_lambda,
			n_hidden_state=FLAGS.n_hidden_state)

		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(judge.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)

		timestamp = '0523_neg_data_char_no_mid_with_attention'
		out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
		print('Writing to {}\n'.format(out_dir))

		loss_summary = tf.summary.scalar('loss', judge.loss)
		acc_summary = tf.summary.scalar('accuracy', judge.accuracy)

		train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
		checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		entity_vocab_processor.save(os.path.join(out_dir, 'entity_vocab'))
		vocab_processor.save(os.path.join(out_dir, 'vocab'))
		entity_class_vocab_processor.save(os.path.join(out_dir, 'entity_class_vocab'))
		pos_vocab_processor.save(os.path.join(out_dir, 'pos_vocab'))

		sess.run(tf.global_variables_initializer())

		def train_step(sentence_batch, entity1_batch, entity2_batch,
			entity1_class_batch, entity2_class_batch, entity1_pos_batch,
			entity2_pos_batch, rel_pos_batch, result_batch):
			feed_dict = {
				judge.input_sentence: sentence_batch,
				judge.entity1_id: entity1_batch,
				judge.entity2_id: entity2_batch,
				judge.entity1_class: entity1_class_batch,
				judge.entity2_class: entity2_class_batch,
				judge.entity1_pos: entity1_pos_batch,
				judge.entity2_pos: entity2_pos_batch,
				judge.relative_pos: rel_pos_batch,
				judge.input_result: result_batch,
				judge.dropout_keep_prob: FLAGS.dropout_keep_prob
			}
			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, judge.loss, judge.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(sentence_batch, entity1_batch, entity2_batch,
			entity1_class_batch, entity2_class_batch, entity1_pos_batch,
			entity2_pos_batch, rel_pos_batch, result_batch, curr_turn, writer=None):
			feed_dict = {
				judge.input_sentence: sentence_batch,
				judge.entity1_id: entity1_batch,
				judge.entity2_id: entity2_batch,
				judge.entity1_class: entity1_class_batch,
				judge.entity2_class: entity2_class_batch,
				judge.entity1_pos: entity1_pos_batch,
				judge.entity2_pos: entity2_pos_batch,
				judge.relative_pos: rel_pos_batch,
				judge.input_result: result_batch,
				judge.dropout_keep_prob: 1.0
			}
			step, summaries, loss, accuracy, predictions, confusion_matrix= sess.run(
				[global_step, dev_summary_op, judge.loss,
				judge.accuracy, judge.predictions, judge.confusion_matrix], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			print("confusion matrix:\n")
			print(confusion_matrix)
			if writer:
				with open('output' + str(curr_turn) + '.txt', 'a') as output_file:
					for i in range(start_index, end_index):
						output_file.write(input_sentences_test[i] + '/' + entity1_names_test.split()[i] + '/' + entity2_names_test.split()[i] + '/')
						if predictions[i % 1600] == 0:
							output_file.write('True/')
						else:
							output_file.write('False/')
						if np.any(input_result_test[i] - [1, 0]) == False:
							output_file.write('Actual:True\n')
						else:
							output_file.write('Actual:False\n')

		batches = data_helpers.batch_iter(
			list(zip(sen_train, e1_train, e2_train, e1_class_train, e2_class_train,# e1_btw_train, e2_btw_train, e3_btw_train,
				e1_pos_train, e2_pos_train, pos_train, res_train)),
				FLAGS.batch_size, FLAGS.num_epochs)

		for batch in batches:
			sen_train, e1_train, e2_train, e1_class_train, e2_class_train, e1_pos_train, e2_pos_train, pos_train, res_train = zip(*batch)
			train_step(sen_train, e1_train, e2_train, e1_class_train, e2_class_train, e1_pos_train, e2_pos_train, pos_train, res_train)
			current_step = tf.train.global_step(sess, global_step)

			if current_step % FLAGS.evaluate_every == 0:
				print('\nValidation:')
				dev_step(sen_val, e1_val, e2_val, e1_class_val, e2_class_val, e1_pos_val, e2_pos_val, pos_val, res_val, current_step)
				
				print('\nTest:')
				for i in range(len(sentence_matrix_test) // 1600 + 1):
					sen_test = sentence_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e1_test = entity1_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e2_test = entity2_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e1_class_test = entity1_class_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e2_class_test = entity2_class_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e1_pos_test = entity1_pos_lot_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					e2_pos_test = entity2_pos_lot_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					pos_test = pos_matrix_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					res_test = input_result_test[i * 1600 : min((i + 1) * 1600, len(sentence_matrix_test))]
					start_index = i * 1600
					end_index = min((i + 1) * 1600, len(sentence_matrix_test))
					dev_step(sen_test, e1_test, e2_test, e1_class_test, e2_class_test, e1_pos_test, e2_pos_test, pos_test, res_test, current_step, writer=True)

			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print('Saved model checkpoint to {}\n'.format(path))
