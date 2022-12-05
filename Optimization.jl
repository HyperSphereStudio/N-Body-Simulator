function newtons_method(f, initial_x, iterations=20, ϵ = 1E-5)
    x = initial_x
    for i in 1:iterations
        f_x = f(x)
        deriv_x = (f(x + ϵ) - f_x) / ϵ
        (deriv_x == 0) && return x
        x -= f_x / deriv_x
    end
    return x
end