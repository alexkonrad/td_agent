## Intro
# include("src/tic-tac-toe.jl")
# using Revise
# using .TicTacToe
# s = State()

include("src/tic-tac-toe.jl")
s = TicTacToe.State()
TicTacToe.is_over(s)

## Ok

s.data[1,:] .= 2
winner = isover(s.data)
