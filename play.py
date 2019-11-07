#!/usr/bin/env python3

#from flask import Flask, Response, request
import time
#from train import Net
#import torch
import chess
import traceback
import chess.svg
from state import State
from random import randint

s = State()

if __name__ == '__main__':

    #self play
    while not s.board.is_game_over():
        #l = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
        options = s.edges()
        choice = randint(0, len(options)-1)
        move = options[choice]
        print(move)
        s.board.push(move)
    print(s.board.result())

