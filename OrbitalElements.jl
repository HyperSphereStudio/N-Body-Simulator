using LinearAlgebra


function orbital_elements(r, v, u = 398600)
	dist = norm(r)
	speed = norm(v)
	radial_v = dot(r, v) / dist
	#radial_v > 0 Away From Perigree
	h = cross(r, v)
	hm = norm(h)
	i = acosd(h[3] / hm)
	#90 < i <= 180 h points in southern direction and is retrograde (opposie earths rotation)
	
	N = cross([0, 0, 1], h)
	Nm = norm(N)
	Ω = acosd(N[1] / Nm)
	Ω = N[2] < 0 ? 360 - Ω : Ω
	
	e = ((speed^2 - u/dist) * r - dist * radial_v * v) / u
	em = norm(e)
	
	ω = acosd(dot(N, e) / (Nm * em))
	ω = e[3] > 0 ? ω : 360 - ω
	
	θ = acosd(dot(e, r) / (em * dist))
	θ = radial_v > 0 ? θ : 360 - θ
	
	return (em, hm, i, Ω, ω, θ)
end
