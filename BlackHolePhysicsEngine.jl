module BlackHoleSimulator

using LinearAlgebra
using Plots
using Reel

include("GeneralizedContextOperations.jl")
@CreateExecutionContext(false)

    
#Universal Gravitational Constant in km^2
#const G = 6.6743E-11 / 1000^2
const G = .1
const ZFVec = ntuple(x -> Float32(0), 3)
const FVec = NTuple{3, Float32}

struct Body
    m
    r
    v

    Body(m, r, v) = new(m, r, v)
end

struct UniverseRef{PIT, MIT}
    positional_information::PIT
    masses::MIT
    ΔT::Float32

    function UniverseRef(u)
        pi = MakeRef(u.positional_information)
        mi = MakeRef(u.masses)
        return new{typeof(pi), typeof(mi)}(pi, mi, Float32(u.ΔT))
    end
    
    function sym_index(s::Symbol)
        if s == :mass
            return 0
        elseif s == :pos
            return 1
        elseif s == :vel
            return 2
        elseif s == :tpos
            return 3
        elseif s == :tvel
            return 4
        elseif s == :com || s == :nacc
            return 5
        end
        throw(error("Unable To Identify Symbol $s"))
    end

    @inline Base.getindex(ur::UniverseRef, s::Symbol, i) = ur[Val{sym_index(s)}(), i]
    @inline Base.setindex!(ur::UniverseRef, v, s::Symbol, i) = ur[Val{sym_index(s)}(), i] = v

    @inline Base.getindex(ur::UniverseRef, ::Val{0}, i) = ur.masses[i]
    @inline Base.setindex!(ur::UniverseRef, m, ::Val{0}, i) = ur.masses[i] = m
    @inline Base.getindex(ur::UniverseRef, ::Val{N}, i) where N = ur.positional_information[i, N]
    @inline Base.setindex!(ur::UniverseRef, v, ::Val{N}, i) where N = ur.positional_information[i, N] = v
end 

mutable struct Universe
    positional_information::TArray{FVec, 2}
    masses::TArray{Float32, 1}
    m::Float64
    com::FVec
    ΔT::Float64
    I::Int

    function Universe(ΔT::Float64, bodies::Body...)
        ba = TArray{FVec, 2}(undef, length(bodies), 5)
        masses = TArray{Float32, 1}(undef, length(bodies))
        for i in eachindex(bodies)
            masses[i] = Float32(bodies[i].m)
            ba[i, 1] = FVec(bodies[i].r)
            ba[i, 2] = FVec(bodies[i].v)
        end
        m = sum(b -> b.m, bodies)
        return new(ba, masses, m, ZFVec, ΔT, 0)
    end
end

function gravitational_kernel(b, r, ΔT)
    #Zero out Net Acceleration
    r[:nacc, b] = ZFVec

    #Net Acceleration
    for b2 in 1:n
        a = r[:pos, b2] .- r[:pos, b]
        d = norm(a)
        F_g = r == 0 ? 0 : G * r[:mass, b2] ./ d^3
        r[:nacc, b] = r[:nacc, b] .+ F_g .* a
    end

    r[:tvel, b] = r[:vel, b] .+ r[:nacc, b] .* ΔT
    r[:tpos, b] = r[:pos, b] .+ r[:tvel, b] .* ΔT

    return nothing
end

function positional_copy_com_calculation_kernel(b)
    #Move temp to actual
    r[:vel, b] = r[:tvel, b]
    r[:pos, b] = r[:tpos, b]

    #Calculate COM
    r[:com, b] = r[:mass, b] .* r[:pos, b]
    return nothing
end

function state_instance(u::Universe, r::UniverseRef)
    n = length(u.masses)
    iterate(gravitational_kernel, 1:n, r, u.ΔT)
    iterate(positional_copy_com_calculation_kernel, 1:n)
    u.com = mapreduce(b -> r[:com, b], .+, 1:n) / u.m
    u.I += 1
end

function simulate(u::Universe, T, state_visitor::Function)
    u.ΔT = step(T)
    ur = UniverseRef(u)

    @gif for t in T
        state_instance(u, ur)
        state_visitor(ur)
    end every 10
end

function run_final(bodies::Body...; T = 0.0:1:200)
    N = length(bodies)
    u = Universe(step(T), bodies...)

    println("Starting Universe! of $N bodies and $(length(T)) iterations")
    numf(n) = round(n, digits=3)

    plt = plot3d(N, xlim = (-500, 500), ylim = (-500, 500), zlim = (-500, 500), legend = false)
    
    v = Array{FVec, 1}(undef, N)
    simulate(u, T, 
            function (r::UniverseRef)
                if u.I % 2 == 0
                    for n in 1:N
                        #Center To COM
                        v[n] = r[:pos, n] .- u.com
                    end
                    scatter(plt, v, color = :jet)
                    println("Finished Iteration $(u.I)")
                end
        end)
end

blackhole = Body(4E3, [0, 0, 0], [0, 0, 0])
bodies = [Body(rand(10:1:400), rand(-50.0:1.0:50.0, 3), rand(-.3:.001:.3, 3)) for j in 1:300]

run_final(blackhole, bodies...)
end