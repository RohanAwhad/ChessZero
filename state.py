#!/usr/bin/env python3
import chess
import numpy as np
import pandas as pd

def state_wise_matrix(bstate, state, threshold, greater_than, less_than, equal):
    for i in range(bstate.shape[0]):
        if greater_than and bstate[i] > threshold: state[i//8][i%8] = bstate[i]-8
        elif less_than and bstate[i] < threshold: state[i//8][i%8] = bstate[i]
        if equal and bstate[i] == threshold: state[i//8][i%8] = bstate[i]
   
    return state

def normalize(state, min_value, max_value):
   for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i][j] > min_value:
                state[i][j] = (state[i][j] - min_value)/(max_value-min_value)

   return state

class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def edges(self):
        return list(self.board.legal_moves)


    def serialize(self):
        assert self.board.is_valid()

        bstate = np.zeros(64, np.uint8)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                bstate[i] = {'P':1, 'N':2, 'B':3, 'R':4, 'Q':5, 'K':6, \
                             'p':9, 'n':10, 'b':11, 'r':12, 'q':13, 'k':14}[pp.symbol()]


        #binary state
        state = np.zeros((6,8,8), np.uint8)
        
        # get black's position
        state[0] = state_wise_matrix(bstate, state[0], 8, True, False, False)
        
        # get white's position
        state[1] = state_wise_matrix(bstate, state[1], 8, False, True, False)

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert bstate[0] == 4
            bstate[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert bstate[7] == 4
            bstate[7] = 7

        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert bstate[56] == 8+4
            bstate[56] = 8+7
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert bstate[63] == 8+4
            bstate[63] = 8+7
        
        
        #get black's castling options
        for i in (14, 15):
            state[2] = state_wise_matrix(bstate, state[2], i, False, False, True)
                
        temp = pd.DataFrame(state[2])
        temp.replace(14, 14-8, inplace=True)
        temp.replace(15, 1, inplace=True)
        state[2] = np.array(temp)

        #get white's castling options
        for i in (6, 7):
            state[3] = state_wise_matrix(bstate, state[3], i, False, False, True)

        temp = pd.DataFrame(state[3])
        temp.replace(7, 1, inplace=True)
        state[3] = np.array(temp)

        if self.board.ep_square is not None:
            assert bstate[self.board.ep_square] == 0
            bstate[self.board.ep_square] = 8
       
        state[4] = state_wise_matrix(bstate, state[4], 8, False, False, True)
        state[4] = state[4]//8 
        
        # 0-3 cols to binary
        #state[0] = (bstate>>3)&1 # Black's position
        #state[1] = (bstate>>2)&1 # castling options 
        #state[2] = (bstate>>1)&1
        #state[3] = (bstate>>0)&1


        # 4th col is who's turn is it
        state[5] = (self.board.turn*1.0)

        # temporary
        print(bstate)
        
        return state


    def shredder_fen_to_vec(x):
        pass

if __name__=='__main__':
    s = State()
    state = s.serialize()
    print('Black Position\n',state[0])
    print('White Position\n',state[1])
    print('Black Castling Options\n',state[2])
    print('White Castling Options\n',state[3])
    print('En passant\n',state[4])
    print('Turn\n',state[5])
