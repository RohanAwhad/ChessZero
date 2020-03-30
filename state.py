#!/usr/bin/env python3

import chess
import numpy as np

class State():
	def __init__(self, player_color, board=None):
		if board == None:
			self.board = chess.Board()
		else:
			self.board = board

		self.player_color = player_color		
		self.M = 12
		self.T = 7
		self.L = 7 # 4 Castling 1 repcounter 1 move number 1 player color
		self.ip = np.zeros((8, 8, self.M * self.T + self.L))
		self.color = [1, 0]
		self.piece_types = ['p', 'n', 'r', 'b', 'q', 'k']

	def serialize_ip(self, move_number):
		assert self.board.is_valid()

		historical_pieces = self.ip[:, :, :-(self.M + self.L)]
		piece_positions = []
		for i in self.color:
			for piece_type in self.piece_types:
				temp = np.zeros((8, 8))
				for index in range(64):
					piece = self.board.piece_at(index)
					if piece is not None and int(piece.color) == i and piece.symbol().lower() == piece_type:
						row = index // 8
						col = index % 8
						temp[row][col] = 1

				temp = np.flip(temp, axis=0) if self.player_color == 0 else temp
				piece_positions.append(temp)

		for i in piece_positions:
			historical_pieces = np.dstack((i, historical_pieces))

		assert historical_pieces.shape == (8, 8, self.M * self.T)

		L_ip = self.ip[:, :, -self.L:]

		# CASTLING SERIALIZATION
		if self.board.has_kingside_castling_rights(chess.WHITE):
			L_ip[:, :, 0] = np.ones((8, 8))
		else:
			L_ip[:, :, 0] = np.zeros((8, 8))

		if self.board.has_queenside_castling_rights(chess.WHITE):
			L_ip[:, :, 1] = np.ones((8, 8))
		else:
			L_ip[:, :, 1] = np.zeros((8, 8))

		if self.board.has_kingside_castling_rights(chess.BLACK):
			L_ip[:, :, 2] = np.ones((8, 8))
		else:
			L_ip[:, :, 2] = np.zeros((8, 8))

		if self.board.has_queenside_castling_rights(chess.BLACK):
			L_ip[:, :, 3] = np.ones((8, 8))
		else:
			L_ip[:, :, 3] = np.zeros((8, 8))

		# REPETITION SERIALIZATION
		rep_cntr = 0
		while self.board.is_repetition(rep_cntr+1):
			rep_cntr += 1
		L_ip[:, :, 4] = np.ones((8, 8)) * rep_cntr

		# MOVE NUMBER
		L_ip[:, :, 5] = np.ones((8, 8)) * move_number

		# PLAYER COLOR
		L_ip[:, :, 6] = np.ones((8, 8)) * self.player_color

		serialized_ip = np.dstack((historical_pieces, L_ip))

		return serialized_ip

	def is_over(self):
		return self.board.is_game_over()

	def score(self):
		assert self.is_over()
		result = self.board.result()
		if len(result) > 3:
			return 0
		else:
			if result[0] == '1':
				return 1
			else:
				return -1

if __name__ == '__main__':
	white = State(1)
	black = State(0, board=white.board)
	#print(white.serialize_ip(0)[:, :, 0])
	#print(black.serialize_ip(0)[:, :, 0])
	black.ip = black.serialize_ip(1)
	for i in range(90):
		print(black.ip[:, :, i])

	print(black.ip[:, :, :12])
