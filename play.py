from game_utils import negamax, negamax1
from model import Magikarp
import chess
import chess.svg
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = {}
config['batch_size'] = 20
config['datafile'] = '../Data/training_data.hdf5'
config['p_datafile'] = '../Data/player_data.hdf5'
config['full_boards_file'] = '../Data/full_boards.pkl'
config['num_epochs'] = 1
config['save_file'] = '../Data/trained_model2/trained_genadv.ckpt'

tf.reset_default_graph()

with tf.Session() as sess:
	# Set up chess board
	board = chess.Board()

	# Load evaluation model
	magikarp = Magikarp(config, sess)
	magikarp.load_model(magikarp.save_file)	

	# Begin chess game
	while not board.is_checkmate():
		# Human plays as white for simplicity
		print('-'*50)
		print("Current Board:\n\n", board, "\n")
		move = "a1a1"

		while True:
			raw_move = input("Please enter a move in UCI notation: ")
			if len(raw_move) != 4:
				print("Please use UCI notation.")
				continue
			move = chess.Move.from_uci(raw_move)
			if move in board.legal_moves:
				board.push(move)
				break
			else:
				print("Please enter a valid move.")

		if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
			print("Congrats - you've won!")
			break
		# Computer response
		score, comp_move = negamax(board, 0, -1, float('-inf'), float('inf'), magikarp)
		print(score, comp_move)
		board.push(comp_move)

		if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
			print("Congrats - you've won!")
			break		

	print('-'*50)
	print("Current Board:\n\n", board, "\n")