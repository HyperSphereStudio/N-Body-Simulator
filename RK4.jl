#There exists an RK-45 Method

"""Numerical Solution to f' = f under the initial conditions f(initial[1]) = initial[2]"""
function rk_4(f::Function, initial, t_f, δ = 1)
    t = initial[1]
    x = initial[2]
    iter = abs(Int(round((t_f - t) / δ)))

    for i in 1:iter
        k1 = f(t, x)
        nt = t + δ/2
        k2 = f(nt, x + δ * k1/2)
        k3 = f(nt, x + δ * k2/2)
        k4 = f(nt, x + δ * k3)
        
        x = x + δ * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = nt
    end

    return x
end

function rk_4_second_order()
    iter = abs(Int(round((t_f - t) / δ)))
    
    for i in 1:iter
        
    end
    
end