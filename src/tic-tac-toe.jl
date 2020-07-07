# TODO: AI Player
#          x Attributes: estimates dictionary
#          x Attributes: epsilon value (probability to explore)
#          x Attributes: step size, states, greedy
#          x Initialize estimates---look up every state in hash table, if
#            current player is the winner then estimate for that state
#            keyed by the hash value is 1, if loser then 0, else 0.5 (and tie).
#          - Update value estimation: calculate TD error for each previous
#            state, then update estimation for state to be the step size times
#            difference in value from next state to last.
#          x Act: Get all possible next states (places the agent can move on the
#            board), then with probability epsilon select the next state randomly,
#            else choose randomly from the highest probability states.
#          - Save/load policy: Save estimations dictionary to file
# TODO: Human player
#          x Implement act function: read keyboard input and
# TODO: Training function
#          - For a number of epochs, make two players play each other.
#          - After each game, call the value estimation update function.
#          - Tally the number of wins by each player, then save the policy
# TODO: Compete function
# TODO: Play function (human player)

module TicTacToe
    using Random: shuffle!
    INDICES = [(1,1) (1,2) (1,3); (2,1) (2,2) (2,3); (3,1) (3,2) (3,3)
        (1,1) (2,1) (3,1); (1,2) (2,2) (3,2); (1,3) (2,3) (3,3)
        (1,1) (2,2) (3,3); (1,3) (2,2) (3,1)]
    KEYS = Dict(
        "q" => (1,1), "w" => (1,2), "e" => (1,3),
        "a" => (2,1), "s" => (2,2), "d" => (2,3),
        "z" => (3,1), "x" => (3,2), "c" => (3,3)
    )
    function board_from_index(i)
        reshape(parse.(Int8, split(lpad(string(i-1, base=3), 9, '0'), "")), 3, 3)
    end
    struct State
        data::Array{Int8,2}
        hash::Int16
        moves1::Vector{Int16}
        moves2::Vector{Int16}
        over::Bool
        winner::Int8
    end
    function legal_moves_from_board(data::Array{Int8,2}, symbol::Int8)
        moves = Int16[]
        for i = 1:3
            for j = 1:3
                if data[i,j] == 0
                    new_board = copy(data)
                    new_board[i,j] = symbol
                    push!(moves, hash(new_board))
                end
            end
        end
        moves
    end
    function is_over(data::Array{Int8,2})
        for (i1, i2, i3) in eachrow(INDICES)
            cells = [data[i1...], data[i2...], data[i3...]]
            if all(x -> x == cells[1] && cells[1] != 0, cells)
                return true, data[i1...]
            end
        end
        return false, -1
    end
    function hash(data::Array{Int8,2})
        # parse(Int, join(string.(vec(data))), base=3)+1
        reduce((acc,x)->3*acc+x, vec(data); init=0)+1
    end
    function State(data::Array{Int8,2}=zeros(Int8,3,3))
        State(data, hash(data),
            legal_moves_from_board(data, Int8(1)),
            legal_moves_from_board(data, Int8(2)),
            is_over(data)...)
    end
    function set_state(cur_state::State, i::Int8, j::Int8, symbol::Int8)
        data = copy(cur_state.data)
        data[i,j] = symbol
        State(data)
    end
    states = map(i -> State(board_from_index(i)), 1:3^9)
    abstract type Player
    end
    mutable struct AIPlayer <: Player
        symbol::Int8
        estimates::Vector{Float64}
        epsilon::Float64
        step_size::Float64
        states::Vector{State}
        greedy::Vector{State}
        function AIPlayer(symbol, epsilon=0.1, step_size=0.1)
            x = new()
            x.symbol = symbol
            x.epsilon = epsilon
            x.step_size = step_size
            x
        end
    end
    struct HumanPlayer <:Player
        symbol::Int8
    end
    function init_estimations(player::Player, states::Vector{State})
        player.estimates = ones(3^9) / 2
        for i in eachindex(states)
            if states[i].over
                player.estimates[i] = states[i].winner == player.symbol ? 1 : 0
            end
        end
    end
    function act(player::AIPlayer, state::State)
        moves = player.symbol == 1 ? state.moves1 : state.moves2
        shuffle!(moves)
        explore = rand() < player.epsilon
        next_state = explore ? states[rand(moves)] : states[moves[argmax(player.estimates[moves])]]
        move_idx = first(findall(state.data .!= next_state.data))
        println("Player $(player.symbol) | Explore: $(explore) | Move: Row $(move_idx[1]), Column $(move_idx[2])")
        return Int8(move_idx[1]), Int8(move_idx[2]), player.symbol
    end
    function act(player::HumanPlayer, state::State)
        println("Your turn. Enter key [qweasdzxc]:")
        key = readline()
        i, j = KEYS[key]
        return (Int8(i), Int8(j), player.symbol)
    end
    mutable struct Game
        p1::Player
        p2::Player
        state::State
        Game() = new(AIPlayer(1), HumanPlayer(2), State())
    end
    function play(game::Game)
        p_iter = Iterators.cycle((game.p1, game.p2))
        N = 0
        while !states[game.state.hash].over
            N += 1
            player, p_iter = Iterators.peel(p_iter)
            i, j, symbol = act(player, game.state)
            game.state = set_state(game.state, i, j, symbol)
            display(game.state.data)
        end
        println("Game over | Winner: Player $(game.state.winner) | $(N) moves.")
    end
    game = Game()
    init_estimations(game.p1, states)
    # init_estimations(game.p2, states)
    play(game)
end
