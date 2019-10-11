using ITensors

 N = 8
sites = spinHalfSites(N)

ampo = AutoMPO(sites)
for j in 1:N-1
    add!(ampo,"Sz",j,"Sz",j+1)
    add!(ampo,0.5,"S+",j,"S-",j+1)
    add!(ampo,0.5,"S+",j,"S-",j+1)
end

H = toMPO(ampo)

psi0 = randomMPS(sites)

sweeps = Sweeps(3)
maxdim!(sweeps,2,2,2)
@show sweeps

energy,psi = dmrg(H,psi0,sweeps)
println("Final energy = $(energy/N)")
