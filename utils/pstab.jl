
function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(P_stab/p.var.rho^2 + P_stab/q.var.rho^2)*(p.x - q.x)
end