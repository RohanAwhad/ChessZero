#!/usr/bin/env python3

from actor_critic import Agent
from state import State
import numpy as np
import sys

class Node():
	def __init__(self, state, policy, player):
		assert state.shape == (1, 8, 8, 91)
		self.state = state # shape (1, 8, 8, 91)
		self.player = player # 1 for white 0 for black
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
	def __init__(self):
		self.reset_board()
		self.agent = Agent()
		self.root = Node(np.expand_dims(self.white_state.serialize_ip(1), axis=0), 1, True)

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
	
	def select_best(self, node):
		best_action = None
		best_ucb = None
		for action, c_node in node.child_nodes.items():
			ucb = c_node.get_UCB(node.N)
			if best_ucb is None or best_ucb < ucb:
				best_ucb = ucb
				best_action = action

		return best_action
			
	
	def simulate_new(self, node, pre_agent_state, post_agent_state, player, move_number):
		assert node.player == player

		value, policy = self.agent.choose_action(np.expand_dims(pre_agent_state.ip, axis=0), pre_agent_state)
		for action, probability in policy:
			pre_agent_state.board.push(action)
			state = np.expand_dims(post_agent_state.serialize_ip(move_number), axis=0)
			node.child_nodes[action] = Node(state, probability, not player)
			pre_agent_state.board.pop()

		self.add_dirichlet_noise(node)
		value = value if player else -value

		return value

	def select_new(self, node):
		move, max_prob = None, 0.0
		for k, v in node.child_nodes.items():
			if max_prob < v.P:
				max_prob = v.P
				move = k

		return move

	def simulate(self, node, move_number):
		value = None
		player = node.player
		for i in range(1600):
			pre_agent_state = self.white_state if player else self.black_state
			post_agent_state = self.black_state if player else self.white_state
			value = self.simulate_new(node, pre_agent_state, post_agent_state, player, move_number)
			if pre_agent_state.is_over():
				value = pre_agent_state.score()
				break
			action = self.select_new(node)
			node = node.child_nodes[action]
			self.white_state.board.push(action)
			self.white_state.ip = self.white_state.serialize_ip(move_number)
			self.black_state.ip = self.black_state.serialize_ip(move_number)
			player = not player
			if player: move_number += 1

		return value

	def traverse(self):
		node = self.root
		visited = [node]
		player = True
		move_number = 1
		self.white_state.ip = self.white_state.serialize_ip(move_number)
		self.black_state.ip = self.black_state.serialize_ip(move_number)
		while len(node.child_nodes) != 0:
			action = self.select_best(node)
			node = node.child_nodes[action]
			self.white_state.board.push(action)
			self.white_state.ip = self.white_state.serialize_ip(move_number)
			self.black_state.ip = self.black_state.serialize_ip(move_number)
			visited.append(node)
			
			print(self.white_state.board)

			player = not player
			if player: move_number += 1
		
		return visited, move_number

	def backpropagate(self, simulation, result):
		simulation.reverse()
		for n in simulation:
			n.N += 1
			n.W += result
			n.Q = n.W / n.N
			result = -result

	def reset_board(self):
		self.white_state = State(1)
		self.black_state = State(0, board=self.white_state.board)

if __name__ == '__main__':
	mct = MonteCarloTreeSearch()
	simulation = []
	cntr = 0
	max_len = 0
	while len(simulation) < 50:
		simulation, move_number = mct.traverse()
		result = mct.simulate(simulation[-1], move_number)
		mct.backpropagate(simulation, result)
		mct.reset_board()
		cntr += 1 
		print(f'Simulation: {cntr} Depth of tree: {len(simulation)}')
		if cntr == 1000: break

