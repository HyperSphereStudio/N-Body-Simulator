#There exists an RK-45 Method

"""Numerical Solution to f' = f under the initial conditions f(initial[1]) = initial[2]"""
function rk_4(f::Function, initial, t_f, δ = 1)
    t = initial[1]
    x = initial[2]
    iter = abs(Int(round((t_f - t) / δ)))

    for i in 1:iter
        k1 = δ * f(t, x)
        k2 = δ * f(t + δ/2, x + k1/2)
        k3 = δ * f(t + δ/2, x + k2/2)
        k4 = δ * f(t + δ, x + k3)
        
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = nt
    end

    return x
end


#rk_4_second_order((t, v, y) -> 9 * v - 20 * y, (0, 1, 4.5), 1.5, .01)
function rk_4_second_order(second_order_diffeq, initial, t_f, δ = 1)
    f = (t, r, v) -> v
    g = second_order_diffeq
    t, r, v = initial
    tv = 0.0
    tr = 0.0

    if t_f < t
        iter = Int(round((t - t_f) / δ))
        δ *= -1
    else
        iter = Int(round((t_f - t) / δ))
    end

    for i in 1:iter
        nt = t
        tr = r
        tv = v

        k1 = δ * f(t, tr, tv)
        l1 = δ * g(t, tr, tv)

        nt = t + δ/2
        tr = r + k1/2
        tv = v + l1/2

        k2 = δ * f(nt, tr, tv)
        l2 = δ * g(nt, tr, tv)

        tr = r + k2/2
        tv = v + l2/2

        k3 = δ * f(nt, tr, tv)
        l3 = δ * g(nt, tr, tv)

        tr = r + k3
        tv = v + l3
        nt = t + δ

        k4 = δ * f(nt, tr, tv)
        l4 = δ * g(nt, tr, tv)
        
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        v += (l1 + 2 * l2 + 2 * l3 + l4) / 6
        t += δ
    end

    return (t, r, v)
end