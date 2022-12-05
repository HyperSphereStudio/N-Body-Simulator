using LinearAlgebra
using Plots

#Universal Gravitational Constant in km^2
const G = 6.6743E-11 / 1000^2


mutable struct Body
    m::Float64
    r::Array{Float64}
    v::Array{Float64}

    nextr::Array{Float64}
    nextv::Array{Float64}

    Body(m, r, v) = new(m, r, v, zeros(Float64, 3), zeros(Float64,3))
end

mutable struct Universe{N}
    bodies::NTuple{N, Body}
    ΔT::Float64
    T::Float64
    I::Int
    m::Float64
    cg::Array{Float64}

    function Universe(bodies::Body...)
        return new{length(bodies)}(bodies, 0, 0, 0, sum(b -> b.m, bodies), zeros(Float64, 3))
    end

    Base.iterate(u::Universe, state = 1) = length(u.bodies) >= state ? (u.bodies[state], state + 1) : nothing
    Base.getindex(u::Universe, i) = u.bodies[i]
    Base.length(::Universe{N}) where N = N
    Base.firstindex(u::Universe) = 1
end

#Acceleration due to gravity of body 2 on body 1
function gravitational_accel(b1::Body, b2::Body)
    rv = b2.r - b1.r
    r = norm(rv)
    F_g = G * b2.m / r^3
    return F_g * rv
end

function net_acceleration(b1, u::Universe)
    net_accel = zeros(Float64, 3)
        for b2 in u
            (b1 == b2) && continue
            net_accel += gravitational_accel(b1, b2)
        end
    return net_accel
end

function state_instance(u::Universe)
    ΔT = u.ΔT
    
    Threads.@threads for b in u
        a = net_acceleration(b, u)
        b.nextv .= b.v .+ a .* ΔT
        b.nextr .= b.r .+ b.nextv .* u.ΔT
    end

    for b in u
        copy!(b.v, b.nextv)
        copy!(b.r, b.nextr)
    end

    u.T += u.ΔT
    u.cg = centerofmass(u)
    u.I += 1
end

function centerofmass(u::Universe)
    firstmoment = zeros(Float64, 3)
    for b in u
        firstmoment += b.m * b.r
    end
    return firstmoment / u.m
end

function simulate(u::Universe, T, state_visitor::Function)
    u.T = first(T)
    u.ΔT = step(T)
    
    @gif for t in T
        state_instance(u)
        state_visitor(u)
    end every 10

end

function run_final(s::Universe{N}, T = 0.0:.1:100) where N
    println("Starting Universe! of $N bodies and $(length(T)) iterations")
    numf(n) = round(n, digits=3)

    plt = plot3d(N)
    
    open("data.dat", "w") do io
        
        function write_dat_header()
            write(io, "t\t\t\t|")
            for i in 1:N
                write(io, "\t\t\tr$i\t\t\t|\t\t\tv$i\t\t\t|")
            end
            write(io, "\n")
        end
        
        function write_dat_entry()
            write(io, "$(numf(s.T))\t\t\t|")
            for b in s
                write(io, "\t$(numf.(b.r))\t|\t$(numf.(b.v))\t|")
            end
            write(io, "\n")
        end
    
        write_dat_header()
        v = Array{Array{Float64}, 1}(undef, N - 1)

        simulate(s, T, 
            function (s::Universe)
                write_dat_entry()
                if s.I % 2 == 0
                    earth = s[1].r
                    for n in 2:N
                        #Center To Earth
                        v[n - 1] = earth - s[n].r
                    end
                    scatter(plt, v, color = :jet)
                    println("Finished Iteration $(s.I)")
                end
            end)
    end

    return display(plt)
end

b1 = Body(4, [2, 1, 3], [65, 1, 7])
b2 = Body(1, [-2, 1, -3], [-65, 1, -7])

run_final(Universe(b1, b2))