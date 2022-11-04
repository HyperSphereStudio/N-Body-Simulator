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
    m::Float64
    cg::Array{Float64}

    function Universe(bodies::Body...)
        return new{length(bodies)}(bodies, 0, 0, sum(b -> b.m, bodies), zeros(Float64, 3))
    end

    Base.iterate(u::Universe, state = 1) = length(u.bodies) >= state ? (u.bodies[state], state + 1) : nothing
    Base.getindex(u::Universe, i) = u.bodies[i]
    Base.length(::Universe{N}) where N = N
end

function gravitational_accel(b1::Body, b2::Body)
    rv = b1.r - b2.r
    r = norm(rv)
    F_g = G * b2.m / r^2
    rn = normalize(rv)
    return -F_g * rn
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
    for b in u
        a = net_acceleration(b, u)
        b.nextv = b.v + a * u.ΔT
        b.nextr = b.r + b.nextv * u.ΔT
    end

    for b in u
        copy!(b.v, b.nextv)
        copy!(b.r, b.nextr)
    end

    u.T += u.ΔT
    u.cg = centerofgravity(u)
end

function centerofgravity(u::Universe)
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
    end every 1

end

function run_final(s::Universe{N}) where N
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
    
        simulate(s, 0.0:100:100000, 
            function (s::Universe)
                write_dat_entry()
                push!(plt, [[s[k].r[i] for k in 1:N] for i in 1:3]...)
            end)
    
    end
end

b1 = Body(5.97E24, [0.0, 0, 0.0], [100, -100, 0])
b2 = Body(7.347E22, [50000, 25000, -50000], [100, -100, 500])

run_final(Universe(b1, b2))
