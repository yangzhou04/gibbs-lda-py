#-*- coding: UTF-8 -*-
'''
Created on 2012-12-26

@author: zhou yang
'''

import os
import re
import glob
import string
import codecs


from util import switch



folder = r'C:\Disk\data\BioNLP-ST_2013_GE_data_sample'
plain_txt = folder + r'\*.txt'
ne_info = folder + r'\*.a1'
event_info = folder + r'\*.a2'
stopwords_path = r'.\stopwords.txt'

def read_stopwords(path):
	'''
	'''
	stopwords = []
	with open(path) as f:
		for word in f.readlines():
			stopwords.append(word[:-1]) #ignore \n

	return stopwords


def spilt_label(line):
	'''
	read start and end index from ne_info_list
	'''
	
	try:
		id_, type_info, name = line[:-1].split('\t', 2)# remove '\n' at the end
		type_, start, end = type_info.split()
	except:
		return ('0', '0', 0, 0, '0') # end of file

	#print 'id_ = %s, type_ = %s, start = %s, end = %s, name = %s'%(id_, type_, start, end, name)
	return (id_, type_, int(start), int(end), name)



def bio_preprocess(stopwords = []):
	'''
	Bio NLP preprocess
	'''


	buf = ''
	# state definition
	NORMAL, NE, PUNC, FINISH = ('NORMAL', 'NE', 'PUNC', 'FINISH')
	SEPERATOR = (',', ';', '.', '\n', ' ', '(', ')', ':')

	ne_info_list = glob.glob(ne_info)
	plain_txt_list = glob.glob(plain_txt)
	ne_word_list = [filename.replace('.txt', '.a3') for filename in plain_txt_list ]

	word_list = []

	# length of ne_info_list and plain_txt_list is the same
	for i in range(len(plain_txt_list)):
		p = codecs.open(plain_txt_list[i], 'r', encoding='utf-8')
		n = codecs.open(ne_info_list[i], 'r', encoding='utf-8')
		w = codecs.open(ne_word_list[i], 'w', encoding='utf-8')

		state = NORMAL
		buf = ''
		pos = -1
		id_, type_, start, end, name = spilt_label(n.readline())


		while state != FINISH:
			ch = p.read(1).lower()
			pos += 1
			buf += ch

			if ch == '':
				state = FINISH

			for case in switch(state):
				if case(NORMAL):
					if pos == start:
						state = NE
					if ch in SEPERATOR:
						buf = buf[:-1] # ignore SEPERATOR
						if buf != '':
							#import pdb; pdb.set_trace()
							if buf.isdigit():
								buf = 'DIGIT'
							if buf not in stopwords and buf not in string.punctuation:
								word_list.append(buf)
								w.write(buf + '\t0\n')
							buf = ''
					break
				if case(NE):
					if pos == end-1:
						if buf not in stopwords:
							word_list.append(buf)
							w.write(buf + '\t1\n')
						buf = ''
						id_, type_, start, end, name = spilt_label(n.readline())
						state = NORMAL
					break
				if case(FINISH):
					print plain_txt_list[i],' processed!'
					break
				if case():
					raise 'Fatal logical error'

		p.close()
		n.close()
		w.close()

	return word_list


def main():
	stopwords = read_stopwords(stopwords_path)
	word_list = bio_preprocess(stopwords)
	print word_list


if __name__ == '__main__':
	main()
	print 'Done!'

