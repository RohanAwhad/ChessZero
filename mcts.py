#!/usr/bin/env python3

from actor_critic import Agent
from state import State
import numpy as np
import sys
from pickle import dump

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
		t = 1 if self.player else -1
		ret = (self.Q * t) + (np.sqrt(parent_N) * c_puct * self.P / (1 + self.N))
		return ret


class MonteCarloTreeSearch():
	def __init__(self):
		self.reset_board()
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
			action = self.select_action(node)
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

	def push_action(self, node, action, move_number):
		self.white_state.board.push(action)

		self.white_state.ip = self.white_state.serialize_ip(move_number)
		self.black_state.ip = self.black_state.serialize_ip(move_number)

		return node.child_nodes[action]

	def self_play(self):
		prev_node = self.root
		player = prev_node.player
		self.reset_board()
		move_number = 1
		self.white_state.ip = self.white_state.serialize_ip(move_number)
		self.black_state.ip = self.black_state.serialize_ip(move_number)
		visited = [] # node, action, ucb
		print(self.white_state.board)
		while True:
			move_number += 1
			prev_node = self.simulate(prev_node, move_number)
			action, ucb = self.select_action(prev_node)
			node = self.push_action(prev_node, action, move_number)
			visited.append((prev_node, action, ucb))
			print(self.white_state.board)
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
		dataset = [] # state, action, ucb, value
		for n, a, p in visited:
			if win_player is None:
				v = 0
			elif n.player == win_player:
				v = 1
			else:
				v = -1
			dataset.append((n.state, a, p, v))
		
		with open('datasets/game_' + str(game) + '.pkl', 'wb') as op:
			dump(dataset, op)


if __name__ == '__main__':
	mct = MonteCarloTreeSearch()
	game_counter = 1
	iteration = 0
	while True:
		mct.self_play(game_counter)
		iteration = Agent.train(iteration)
		game_counter += 1
