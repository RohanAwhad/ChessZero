#!/usr/bin/env python3

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, Flatten, Add
from keras.models import Model
from keras.optimizers import Adam
from state import State
import numpy as np

class ActorCritic():
	def __init__(self, depth_size):
		self.build_model(depth_size)		
		self.compile()

	def compile(self):
		losses = {'value_op': 'mean_squared_error', 'policy_op': 'categorical_crossentropy'}
		loss_weights={'value_op': 1.0, 'policy_op': 1.0}
		optimizer = Adam(lr=0.2)
		self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

	def get_lr(self, no_of_steps):
		if no_of_steps < 100000:
			lr = 0.2
		elif no_of_steps < 300000:
			lr = 0.02
		elif no_of_steps < 500000:
			lr = 0.002
		else:
			lr = 0.0002

		config = self.model.optimizer.get_config()
		config['lr'] = lr
		self.model.optimizer.from_config(config)

	def nn_body(self, ip):
		x = self.conv_block(ip)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		for _ in range(19):
			x = self.identity_res_block(x)

		return x

	def value_head(self, ip):
		x = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(ip)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Flatten()(x)
		x = Dense(256, activation='relu')(x)
		x = Dense(1, activation='tanh', name='value_op')(x)
		return x


	def policy_head(self, ip):
		x = self.conv_block(ip)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Conv2D(filters=73, kernel_size=(3, 3), padding='same', activation='softmax', name='policy_op')(x)
		return x

	def conv_block(self, ip):
		return Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(ip)

	def identity_res_block(self, ip):
		conv1 = self.conv_block(ip)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)

		conv2 = self.conv_block(conv1)
		conv2 = BatchNormalization()(conv2)

		res = Add()([conv2, ip])
		return Activation('relu')(res)

	def build_model(self, depth_size):
		inputs = Input(shape=(8, 8, depth_size))
		body = self.nn_body(inputs)
		value_op = self.value_head(body)
		policy_op = self.policy_head(body)

		self.model = Model(inputs=inputs, outputs=[value_op, policy_op])

	def predict(self, ip):
		assert ip.shape == (1, 8, 8, 91)
		predictions = self.model.predict(ip)
		value , policy = np.squeeze(predictions[0], axis=0), np.squeeze(predictions[1], axis=0)
		return (value, policy)

class Agent():
	def __init__(self):
		self.nn = ActorCritic(91)

	def board_rep(self, ip):
		temp = np.zeros((8,8))
		for i in range(12):
			temp += ip[:, :, i]
		return temp

	def get_plane(self, sub_r, sub_c, promo_fl, promo_to):
		if sub_r == 0:
			if sub_c < 0:
				return 20 + abs(sub_c)
			else:
				return 13 + sub_c
		elif sub_c == 0:
			if sub_r < 0 :
				return abs(sub_r)
			else:
				if promo_fl:
					if promo_to == 'n' : return 64
					elif promo_to == 'b' : return 67
					elif promo_to == 'r' : return 70
				return 7 + sub_r
		elif abs(sub_r) == abs(sub_c):
			if sub_r > 0 and sub_c < 0: return 27 + sub_r
			if sub_r < 0 and sub_c < 0: return 34 + abs(sub_r)
			if sub_r > 0 and sub_c > 0: return 41 + sub_r
			if sub_r < 0 and sub_c > 0: return 48 + abs(sub_r)
		else:
			if sub_c == -1:
				if sub_r == 2: return 56
				if sub_r == -2: return 58
			if sub_c == -2:
				if sub_r == 1: return 57
				if sub_r == -1: return 59
			if sub_c == 1:
				if sub_r == 2: return 60
				if sub_r == -2: return 62
			if sub_c == 2:
				if sub_r == 1: return 61
				if sub_r == -1: return 63

	def get_location(self, move, player_color):

		#without agent_board
		move = move.uci()
		flip_flag = True if player_color == 0 else False

		pickup_col = ord(move[0]) - ord('a')
		pickup_row = int(move[1]) - 1

		post_col = ord(move[2]) - ord('a')
		post_row = int(move[3]) - 1

		if flip_flag:
			post_row = 7 - post_row
			pickup_row = 7 - pickup_row

		sub_row = post_row - pickup_row
		sub_col = post_col - pickup_col

		if len(move) == 5:
			promo_to = move[4].lower()
			promo_fl = True
		else:
			promo_to = None
			promo_fl = False

		depth = self.get_plane(sub_row, sub_col, promo_fl, promo_to)

		return (pickup_row, pickup_col, depth) 

	def serialize_op(self, policy, agent_board, ip_rep):
		legal_moves = list(agent_board.board.legal_moves)
		locations = [(m, self.get_location(m, agent_board.player_color)) for m in legal_moves]
		ret = np.zeros((policy.shape))
		for _, l in locations:
			r, c, d = l
			ret[r, c, d] = policy[r, c, d]
		
		sum_ret = np.sum(ret)
		policy = ret / sum_ret
		
		ret = []
		for m , l in locations:
			r, c, d = l
			ret.append((m, policy[r, c, d]))

		return ret

	def choose_action(self, ip, agent_board):
		value, policy = self.nn.predict(ip)
		ip_rep = self.board_rep(np.squeeze(ip, axis=0))
		serialized_pred = self.serialize_op(policy, agent_board, ip_rep)
		return value, serialized_pred

	def train(self, iteration):
		dataset = []
		folder = f'datasets/iteration_{iteration}/'
		data_files = os.listdir(folder)
		
		if len(data_files < 10):
			return 

		for f in data_files:
			with open(folder+f, 'rb') as ip:
				temp_data = load(ip)
			dataset.extend(temp_data)

		# dataset = [(state, action, policy, value)]
		X = np.zeros((4096, 8, 8, 91))
		Y_policy = np.zeros((4096, 8, 8, 73))
		Y_value = np.zeros((4096, 1))
		for i in range(len(dataset)):
			state, action, policy, value = dataset[i]
			X[i] = np.squeeze(state)
			r, c, d = self.get_location(action, X[i, 0, 0, -1])
			Y_policy[i, r, c, d] = policy
			Y_value[i] = value

		self.nn.fit(X, {'value_op' : Y_value, 'policy_op' : Y_policy})

if __name__ == '__main__':
	agent = Agent()
	state = State(1)
	agent.choose_action(np.expand_dims(state.ip, axis=0), state)
