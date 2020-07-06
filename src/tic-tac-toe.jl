# TODO: AI Player
#          - Attributes: estimates dictionary
#          - Attributes: epsilon value (probability to explore)
#          - Attributes: step size, states, greedy
#          - Initialize estimates---look up every state in hash table, if
#            current player is the winner then estimate for that state
#            keyed by the hash value is 1, if loser then 0, else 0.5 (and tie).
#          - Update value estimation: calculate TD error for each previous
#            state, then update estimation for state to be the step size times
#            difference in value from next state to last.
#          - Act: Get all possible next states (places the agent can move on the
#            board), then with probability epsilon select the next state randomly,
#            else choose randomly from the highest probability states.
#          - Save/load policy: Save estimations dictionary to file
# TODO: Human player
#          - Implement act function: read keyboard input and
# TODO: Training function
#          - For a number of epochs, make two players play each other.
#          - After each game, call the value estimation update function.
#          - Tally the number of wins by each player, then save the policy
# TODO: Compete function
# TODO: Play function (human player)

module TicTacToe
    using LinearAlgebra: Diagonal

    struct State
        data::Array{Int8,2}
        hash::Int16
        over::Bool
        winner::Int8
    end

    function board_from_index(i)
        reshape(parse.(Int, split(lpad(string(i-1, base=3), 9, '0'), "")), 3, 3)
    end

    function is_over(data::Array{Int8,2})
        winning = all_lines(data) |>
            sums->filter(!iszero, sums) |>
            sums->filter(x->x%3==0, sums) |>
            x -> div.(x, 3)
        if length(winning) == 0
            return false, -1
        end
        return true, winning[1]
    end

    function hash(data::Array{Int8,2})
        parse(Int, join(string.(vec(data))), base=3)+1
        reduce((acc,x)->3*acc+x, vec(data); init=0)+1
    end

    function all_lines(data::Array{Int8,2})
        vcat(sum(data, dims=1)', sum(data, dims=2),
            sum(Diagonal(data)), sum(Diagonal(reverse(data, dims=2))))
    end

    function State(data::Array{Int8,2}=zeros(Int8,3,3))
        State(data, hash(data), is_over(data)...)
    end

    function emptycell_indices(state::State)
        indices = Iterators.product(UnitRange{Int8}(1,3), UnitRange{Int8}(1,3))
        emptycells(i) = Iterators.filter(x -> iszero(state.data[x...]), i)
        indices |> emptycells
    end

    function next_state!(current_state, current_symbol, all_states)
        for (i,j) in emptycell_indices(current_state)
            new_state = set_state(current_state, i, j, current_symbol)
            if !isassigned(all_states, new_state.hash)
                all_states[new_state.hash] = new_state
                next_symbol::Int8 = current_symbol == 1 ? 2 : 1
                next_state!(new_state, next_symbol, all_states)
            end
        end
    end

    function set_state(cur_state::State, i::Int8, j::Int8, symbol::Int8)
        data = copy(cur_state.data)
        data[i,j] = symbol
        State(data)
    end

    function get_states()
        all = Vector{State}(undef, 3^9)
        symbol::Int8 = 1
        state = State()
        all[state.hash] = state
        next_state!(state, symbol, all)
        all
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
        AIPlayer(epsilon=0.1, step_size=0.1) = (x = new(); x.epsilon = epsilon; x.step_size = step_size; x)
    end
    struct HumanPlayer <:Player
        symbol::Int8
    end
    function init_estimations(player::Player, states::Vector{State})
        player.estimates = ones(3^9) / 2
        # final_states = filter((hash, S) -> S.over, states)
        println(states[1:100])
        for i in eachindex(states[1:100])
            if isassigned(states, i)
                println(states[i])
            end
        end
        # winning_idx = [state.winner == player.symbol for state in states]
        # println(winning_idx)
        # for (hash, state) in final_states
        #     println(state.over)
        # end
        # for i in 0:3^9
        #     println(i in states.keys)
        #     if states.i.over == true
        #         # TODO: do something for stalemate
        #         if states[i].winner == player.symbol
        #             player.estimates[i] = 1
        #         elseif states[i].winner != player.symbol
        #             player.estimates[i] = 0
        #         end
        #     end
        # end
    end
    function act(player::Player, state::State)
        return Int8(1), Int8(1), Int8(2)
    end
    struct Game
        p1::Player
        p2::Player
        state::State
        Game() = new(AIPlayer(), AIPlayer(), State())
    end
    function play(game::Game)
        p_iter = Iterators.cycle((game.p1, game.p2))
        while !states[game.state.hash].over
            player, p_iter = Iterators.peel(p_iter)
            i, j, symbol = act(player, game.state)
            # game.state = set_state(game.state, i, j, symbol)
        end
    end
    game = Game()
    init_estimations(game.p1, states)
    init_estimations(game.p2, states)
    play(game)
end
