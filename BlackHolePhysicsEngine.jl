using LinearAlgebra
using Plots
using Reel

include("GeneralizedContextOperations.jl")
@CreateExecutionContext(true)

#Universal Gravitational Constant in km^2
#const G = 6.6743E-11 / 1000^2
const G = .1
const FVec = TStkArray{3, FP}

struct Body
    m::Float64
    r::FVec
    v::FVec

    nextr::FVec
    nextv::FVec
    nexta::FVec

    Body(m, r, v) = new(m, r, v, zeros(FVec), zeros(FVec), zeros(FVec))
end

@RegisterType(Body)

struct Universe
    bodies::TRef{TArray{Body, 1}}
    m::Float64
    cg::TRef{FVec}
    ΔT::Float64

    function Universe(ΔT::Float64, bodies::Body...)
        ba = TArray{Body, 1}(undef, length(bodies))
        copyto!(ba, bodies)
        m = sum(b -> b.m, bodies)
        return new(TRef(ba), m, TRef(zeros(FVec)), ΔT)
    end

    Base.getindex(u::Universe, i) = u.bodies[i]
    Base.length(u::Universe) = length(u.bodies)
end

@RegisterType(Universe)

function state_instance(u::Universe)
    ΔT = u.ΔT

    linearforeach(function ()
            body_1 = u[][b1]

            #Net Acceleration
            for b2 in 1:length(u)
                body_2 = v[][b2]
        
                #Use as Temp
                a = body_1.nextr
                a .= body_2.r .- body_1.r
                r = norm(a)
                F_g = r == 0 ? 0 : G * mass(v, b2) / r^3
                a *= F_g
        
                body_1.nexta += a
            end

            a = body_1.nexta
            b.nextv .= b.v .+ a .* ΔT
            b.nextr .= b.r .+ b.nextv .* u.ΔT

            return body_1
        end, u.bodies[], u.bodies[])

    for b in u
        b.v .= b.nextv
        b.r .= b.nextr
    end

    firstmoment = u[].cg
    firstmoment .= 0
    for b in u[]
        firstmoment += b.m * b.r
    end
    firstmoment ./ u.m
end

function simulate(u::TRef{Universe}, T, state_visitor::Function)
    u[].ΔT = step(T)
    
    @gif for t in T
        state_instance(u)
        state_visitor(u)
    end every 10

end

function run_final(bodies::Body...; T = 0.0:1:200)
    N = length(bodies)
    u = TRef(Universe(step(T), bodies...))

    println("Starting Universe! of $N bodies and $(length(T)) iterations")
    numf(n) = round(n, digits=3)

    plt = plot3d(N, xlim = (-500, 500), ylim = (-500, 500), zlim = (-500, 500), legend = false)
    
    v = Array{NTuple{3, Float64}, 1}(undef, N)
    simulate(u, T, 
            function (s::Universe)
                if s.I % 2 == 0
                    for n in 1:N
                        #Center To CG
                        v[n] = ntuple(d -> s[n].r[d] - s.cg[d], 3)
                    end
                    scatter(plt, v,  color = :jet)
                    println("Finished Iteration $(s.I)")
                end
        end)
end

blackhole = Body(4E3, zeros(FVec), zeros(FVec))
bodies = [Body(rand(10:1:400), rand(-50.0:1.0:50.0, 3), rand(-.3:.001:.3, 3)) for j in 1:300]

run_final(blackhole, bodies...)