#!/usr/bin/env python3

from actor_critic import Agent
from state import State
import numpy as np
import sys
from pickle import dump
import os

class Node():
	def __init__(self, state, parent, policy, player):
		assert state.shape == (1, 8, 8, 91)
		self.state = state # shape (1, 8, 8, 91)
		self.player = player # 1 for white 0 for black
		self.parent = parent
		self.N = 0
		self.W = 0
		self.Q = 0
		self.P = policy
		self.child_nodes = dict() # action: child_node

	def get_UCB(self, parent_N):
		c_puct = 4
		ret = self.Q + (np.sqrt(parent_N) * c_puct * self.P / (1 + self.N))
		return ret


class MonteCarloTreeSearch():
	def __init__(self, agent=None):
		self.reset_board()
		if agent is not None:
			self.agent = agent
		else:
			self.agent = Agent()
		self.root = Node(np.expand_dims(self.white_state.serialize_ip(1), axis=0), None, 1, True) 

	def add_dirichlet_noise(self, node):
		child_nodes = []
		for k, v in node.child_nodes.items():
			child_nodes.append(v.P)

		child_nodes = np.array(child_nodes)
		noised_cn = 0.75*child_nodes + 0.25*np.random.dirichlet(np.zeros(len(child_nodes), dtype=np.float32)+192) # from alphazero connect 4 code

		cntr = 0
		for _, v in node.child_nodes.items():
			v.P = noised_cn[cntr]
			cntr += 1

	def select_action(self, node):
		best_action = None
		best_ucb = None
		for action, c_node in node.child_nodes.items():
			ucb = c_node.get_UCB(node.N)
			if best_ucb is None or best_ucb < ucb:
				best_ucb = ucb
				best_action = action

		return best_action, best_ucb

	def backup(self, node, result):
		while True:
			node.N += 1
			node.W += result
			node.Q = node.W / node.N
			result = -result
			if node.parent is None:
				break
			node = node.parent

	def reset_board(self):
		self.white_state = State(1)
		self.black_state = State(0, board=self.white_state.board)
			
	def traverse(self, node, move_number):
		player = node.player
		visited = []
		game_over_flag = False
		while len(node.child_nodes) != 0:
			action, _ = self.select_action(node)
			node = self.push_action(node, action, move_number)
			visited.append(node)
			move_number += 1
			player = not player
			if self.white_state.is_over():
				game_over_flag = True
				break

		if not game_over_flag:
			pre_agent_state = self.white_state if player else self.black_state
			post_agent_state = self.black_state if player else self.white_state
			value = self.simulate_new(node, pre_agent_state, post_agent_state, player, move_number)
		else:
			value = self.white_state.score()
			if value != 0:
				if value == -1:
					assert player == False
				else:
					assert player == True

		return node, value, len(visited)
			
	def simulate_new(self, node, pre_agent_state, post_agent_state, player, move_number):
		assert node.player == player

		value, policy = self.agent.choose_action(np.expand_dims(pre_agent_state.ip, axis=0), pre_agent_state)
		for action, probability in policy:
			pre_agent_state.board.push(action)
			state = np.expand_dims(post_agent_state.serialize_ip(move_number), axis=0)
			node.child_nodes[action] = Node(state, node, probability, not player)
			pre_agent_state.board.pop()

		self.add_dirichlet_noise(node)

		return value

	def simulate(self, prev_node, move_number):
		root = prev_node
		white_old_ip = self.white_state.ip
		black_old_ip = self.black_state.ip

		for i in range(1600):
			node, result, pop_len = self.traverse(root, move_number)
			self.backup(node, result)
			
			for i in range(pop_len):
				self.white_state.board.pop()

			self.white_state.ip = white_old_ip
			self.black_state.ip = black_old_ip

		return root

	def push_action(self, node, action, move_number, player=None):
		self.white_state.board.push(action)

		self.white_state.ip = self.white_state.serialize_ip(move_number)
		self.black_state.ip = self.black_state.serialize_ip(move_number)

		if action not in node.child_nodes.keys():
			if player:
				state = self.white_state.ip
			else:
				state = self.black_state.ip

			node.child_nodes[action] = Node(state, node, 0, player)

		return node.child_nodes[action]

	def self_play(self, game, iteration):
		prev_node = self.root
		player = prev_node.player
		self.reset_board()
		move_number = 1
		self.white_state.ip = self.white_state.serialize_ip(move_number)
		self.black_state.ip = self.black_state.serialize_ip(move_number)
		visited = [] # node, action, ucb, board
		print(self.white_state.board)
		while True:
			move_number += 1
			prev_node = self.simulate(prev_node, move_number)
			action, ucb = self.select_action(prev_node)
			visited.append((prev_node, action, ucb))
			node = self.push_action(prev_node, action, move_number)
			print(self.white_state.board)
			for i in range(12):
				print(self.white_state.ip[:, :, i])
			for i in range(12):
				print(self.black_state.ip[:, :, i])
			if self.white_state.is_over():
				value = self.white_state.score()
				if value != 0 :
					win_player = True if value > 0 else False
				else:
					win_player = None
				break
			player = not player
			assert node.player == player
			prev_node = node


		# adding value to dataset and saving
		dataset = [] # state, action, q-ucb, value, board
		for n, a, p, b in visited:
			if win_player is None:
				v = 0
			elif n.player == win_player:
				v = 1
			else:
				v = -1
			p -= n.child_nodes[a].Q
			dataset.append((n.state, a, p, v, b))
		
		if not os.path.exists(f'datasets/iteration_{iteration}'):
			os.makedirs(f'datasets/iteration_{iteration}')
		with open('datasets/iteration_' + str(iteration) + '/game_' + str(game) + '.pkl', 'wb') as op:
			dump(dataset, op)


def evaluate(current_iteration, current_agent):

	prev_iter_agent = Agent()
	prev_iter_agent.load_model(current_iteration-1)

	win_rate = 0
	for _ in range(20):

		#choose black or white
		if np.random.randint(0,2):
			# if 1 current_agent = white
			current_agent_color = 1
			prev_iter_color = 0
		else:
			current_agent_color = 0
			prev_iter_color = 1

		mct_c = MonteCarloTreeSearch(current_agent)
		mct_p = MonteCarloTreeSearch(prev_iter_agent)


		if current_agent_color == 1:
			player = mct_c
		else:
			player = mct_p

		move_number = 1
		
		mct_c.white_state.ip = mct_c.white_state.serialize_ip(move_number)
		mct_c.black_state.ip = mct_c.black_state.serialize_ip(move_number)
		mct_p.white_state.ip = mct_p.white_state.serialize_ip(move_number)
		mct_p.black_state.ip = mct_p.black_state.serialize_ip(move_number)
		
		print(player.white_state.board)

		prev_node = player.root
		while True:
			move_number += 1
			prev_node = player.simulate(prev_node, move_number)
			action, ucb = player.select_action(prev_node)
			
			if player == mct_c:
				color = current_agent_color
			else:
				color = prev_iter_color
			c_node = mct_c.push_action(prev_node, action, move_number, bool(color))
			p_node = mct_p.push_action(prev_node, action, move_number, bool(color))
			
			print(player.white_state.board)
			
			if player.white_state.is_over():
				value = player.white_state.score()
				if value != 0 :
					win_player = True if value > 0 else False
				else:
					win_player = None
				break

			if player == mct_c:
				player = mct_p
				prev_node = p_node
			else:
				player == mct_c
				prev_node = c_node

		if win_player is not None :
			if win_player and current_agent_color == 1: win_rate += 1
			elif not win_player and current_agent_color == 0: win_rate += 1

	if win_rate >= 11:
		current_agent.save(current_iteration)
		current_iteration += 1

	return current_iteration


if __name__ == '__main__':
	mct = MonteCarloTreeSearch()
	game_counter = 1
	iteration = 0
	while True:
		mct.self_play(game_counter, iteration)
		mct.agent.train(current_iteration)
		if current_iteration > 0: 
			iteration = evaluate(current_iteration, mct.agent) # evaluates current model and saves it if conditin is met and returns iteration+1
		game_counter += 1
		if game_counter == 150 :
			mct.agent.save(iteration)
			iteration += 1

		'''
		I need to update iteration
		iteration 0 150 games to train and move to iteration 1
		from iteration 1 
			save the model when iteration shifted
			train new model after every game
			evaluate how good it is by playing 20 games with latest saved model(i.e previous generation)
			if current model won min 11 games save the current model 
			iteration += 1

		'''
