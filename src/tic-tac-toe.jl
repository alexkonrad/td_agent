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
    using JLD
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
        if all(x -> x != 0, data)
            return true, -1
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
        greedy::Vector{Bool}
        function AIPlayer(symbol, epsilon=0.1, step_size=0.1)
            x = new()
            x.symbol = symbol
            x.epsilon = epsilon
            x.step_size = step_size
            x.states = State[]
            x.greedy = Bool[]
            x
        end
    end
    struct HumanPlayer <:Player
        symbol::Int8
    end
    function set_player_state!(player::Player, state::State)
        if isa(player, AIPlayer)
            push!(player.states, state)
        end
    end
    function init_estimates!(player::Player, states::Vector{State})
        player.estimates = ones(3^9) / 2
        for i in eachindex(states)
            if states[i].over
                if states[i].winner == player.symbol
                    player.estimates[i] = 1
                elseif states[i].winner > 0
                    player.estimates[i] = 0
                end
            end
        end
    end
    function update_estimates!(player::Player)
        for i in reverse(eachindex(player.states[1:end-1]))
            state = player.states[i]
            next_state = player.states[i+1]
            state_val_estim = player.estimates[state.hash]
            next_val_estim = player.estimates[next_state.hash]
            val_diff = next_val_estim - state_val_estim
            player.estimates[state.hash] += player.step_size * val_diff
        end
    end
    function reset!(player::Player)
        player.states = State[]
    end
    function load_policy!(player::Player)
        p = load("policy.jld")
        if player.symbol == 1
            player.estimates = d.p1
        elseif player.symbol == 2
            player.estimates = d.p2
        end
    end
    function act(player::AIPlayer, state::State)
        moves = player.symbol == 1 ? state.moves1 : state.moves2
        shuffle!(moves)
        explore = rand() < player.epsilon
        next_state = explore ? states[rand(moves)] : states[moves[argmax(player.estimates[moves])]]
        move_idx = first(findall(state.data .!= next_state.data))
        # println("Player $(player.symbol) | Explore: $(explore) | Move: Row $(move_idx[1]), Column $(move_idx[2])")
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
        function Game(human::Bool)
            game = new()
            if human
                game.p1 = HumanPlayer(Int8(1))
            else
                game.p1 = AIPlayer(Int8(1))
                init_estimates!(game.p1, states)
            end
            game.p2 = AIPlayer(Int8(2))
            init_estimates!(game.p2, states)
            game.state = State()
            game
        end
    end
    function train(epochs=1e2)
        println("Training...")
        player1_wins = 0
        player2_wins = 0
        for i in 1:epochs
            game = Game(false)
            winner = play(game)
            if winner == 1
                player1_wins += 1
            elseif winner == 2
                player2_wins += 1
            end
            if i % 50 == 0
                player1_winrate = round(player1_wins / i, sigdigits=2)
                player2_winrate = round(player2_wins / i, sigdigits=2)
                println("Epoch $(i):\t",
                    "Player 1 Win-rate: $(player1_winrate)\t",
                    "Player 2 Win-rate: $(player2_winrate)")
            end
            update_estimates!(game.p1)
            update_estimates!(game.p2)
            reset!(game.p1)
            reset!(game.p2)
            save("policy.jld", "p1", game.p1.estimates, "p2", game.p2.estimates)
        end
    end
    function compete(turns)
        game = Game()
        load_policy!(game.p1)
        load_policy!(game.p2)
        player1_win = 0
        player2_win = 0
        for i in 1:turns
            winner = play(game)
            if winner == 1
                player1_win += 1
            elseif winner == 2
                player2_win += 1
            end
        end
        player1_winrate = round(player1_win / turns, 2)
        player2_winrate = round(player2_win / turns, 2)
        println("$(turns) Turns, ",
            "Player 1 winrate $(player1_winrate), ",
            "Player 2 winrate $(player2_winrate)")
    end
    function play(game::Game)
        p_iter = Iterators.cycle((game.p1, game.p2))
        N = 0
        while !game.state.over
            N += 1
            player, p_iter = Iterators.peel(p_iter)
            set_player_state!(player, game.state)
            i, j, symbol = act(player, game.state)
            game.state = set_state(game.state, i, j, symbol)
            # display(game.state.data)
        end
        # println("Game over | Winner: Player $(game.state.winner) | $(N) moves.")
        return game.state.winner
    end
    train(1e5)
end
