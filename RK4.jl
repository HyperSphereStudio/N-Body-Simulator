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
        
        x += δ * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = nt
    end

    return x
end


#rk_4_second_order((t, v, y) -> 9 * v - 20 * y, (0, 1, 4.5), 1.5, .01)
function rk_4_second_order(accel, initial, r_f, δ = 1)
    f = accel
    g = (t, r) -> r 
    t, r, z = initial
    iter = abs(Int(round((r_f - t) / δ)))

    for i in 1:iter
        nt = t + δ/2

        k1 = f(t, z, r)
        l1 = g(t, r)

        k2 = f(nt, z + δ * k1/2, r)
        l2 = g(nt, r + δ * l1/2)

        k3 = f(nt, z + δ * k2/2, r)
        l3 = g(nt, r + δ * l2/2)

        k4 = f(nt, z + δ * k3, r)
        l4 = g(nt, r + δ * l3)
        
        z += δ * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        r += δ * (l1 + 2 * l2 + 2 * l3 + l4) / 6
        
        t = nt
    end

    return (t, r, z)
end