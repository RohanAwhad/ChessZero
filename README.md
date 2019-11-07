# Chess Zero

### Neural Network based Chess engine with reinforcement learning.
### Toy implementation of AlphaZero

## Major Steps
 1. Figure out how to make chess play with itself
 2. Find how to implement Q Learning (SentDex)
 3.

## States
 1. Black's Position {0: Empty, 1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King}
 2. White's Position {0: Empty, 1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King}
 3. Black's Castling Options {0: Nope, 1: Available Rooks, 14: Black's King}
 4. White's Castling Options {0: Nope, 1: Available Rooks, 6: White's King}
 5. En Passant Move {0: Nope, 1: Yup}
 6. Turn {0: Black, 1: White}
