import os
import tensorflow as tf
import numpy as np
import utils

def DisplayAll(file_path):
	f = open(file_path)
	success_time = 0
	success_count = 0
	success_query = 0
	success_time_list = []
	success_query_list = []
	all_count = 0
	fail_list = []
	while True:
		line = f.readline()
		if not line:
			break
		# 0_success_./100/103.jpg_9.99997395563765_7030_113.398609161
		
		results = line.split('_')
		if len(results) < 6:
			continue
		all_count += 1
		if results[1] == 'success':
			success_count+=1
			success_time += float(results[-1])
			success_query += int(results[-2])
			success_time_list.append(float(results[-1]))
			success_query_list.append(int(results[-2]))
			if float(results[-1]) > 1800:
				fail_list.append(results[0])
		else:
			fail_list.append(results[0])
	success_time_list.sort()
	success_query_list.sort()
	print(success_time_list)
	print(success_query_list)
	print("success rate:{}/{}".format(success_count, all_count))
	print("avr success_time:{}, median:{}".format(success_time/success_count, success_time_list[int(success_count/2)]))
	print("avr success_query:{}, median:{}".format(success_query/success_count, success_query_list[int(success_count/2)]))
	print(fail_list)

	f.close()
	# success_query = 0
	# success_time = 0
	# success_count = 3
	# print("success_time_list[-1]:{}, success_time_list[{}]:{}".format(success_time_list[-1], success_count, success_time_list[success_count]))
	# for i in range(0, success_count):
	# 	success_query += success_query_list[i]
	# 	success_time += success_time_list[i]
	# print("avr success_time:{}, median:{}".format(success_time/success_count, success_time_list[success_count/2]))
	# print("avr success_query:{}, median:{}".format(success_query/success_count, success_query_list[success_count/2]))


def max_abs(perturbation):
	max_value = 0
	for i in range(perturbation.shape[0]):
		for j in range(perturbation.shape[1]):
			for k in range(perturbation.shape[2]):
				if abs(perturbation[i][j][k]) > max_value:
					max_value = abs(perturbation[i][j][k])
	return max_value

def Verify(folder_path):
	ori_root = './100/'
	imgs_set = os.listdir(folder_path)
	imgs_set.sort()
	model,_,_,_ = utils.load_model(3)
	fail_count = 0
	for i in range(len(imgs_set)):
		file_name = imgs_set[i].split('.')[0]
		ori = utils.img_to_input(ori_root+file_name+'.jpg')
		opred = utils.input_to_prediction(np.expand_dims(ori, axis=0),model)
		adv = utils.img_to_input(folder_path+file_name+'.png')
		apred = utils.input_to_prediction(np.expand_dims(adv, axis=0),model)
		perturbation = ori-adv
		max_value = max_abs(perturbation)
		olabel = np.argmax(opred)
		alabel = np.argmax(apred)
		tlabel = utils.find_k_max(opred,10)
		print(olabel, alabel, tlabel, max_value)
		if not alabel == tlabel:
			fail_count += 1
		elif max_value > 10:
			fail_count += 1
	print('fail_count', fail_count)



flags = tf.app.flags
flags.DEFINE_string('cmd', 'all', 'time or query or all.')
flags.DEFINE_string('folder', './log/', 'log folder path.')
flags.DEFINE_string('file', '', 'log file name.')

FLAGS = flags.FLAGS
file_path = FLAGS.folder+FLAGS.file
if FLAGS.cmd == 'time':
	DisplayTime(file_path)
elif FLAGS.cmd == 'query':
	DisplayQuery(file_path)
elif FLAGS.cmd == 'all':
	DisplayAll(file_path)
elif FLAGS.cmd == 'verify':
	Verify('./Results/'+FLAGS.folder+'/success/')

