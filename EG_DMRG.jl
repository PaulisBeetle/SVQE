using ITensors

let
  # Create 100 spin-one (dimension 3) indices
  N = 8
  sites = spinHalfSites(N)

  # Input operator terms which define
  # a Hamiltonian matrix, and convert
  # these terms to an MPO tensor network
  ampo = AutoMPO(sites)
  for j=1:N-1
    add!(ampo,"Sz",j,"Sz",j+1)
    add!(ampo,0.5,"S+",j,"S-",j+1)
    add!(ampo,0.5,"S-",j,"S+",j+1)
  end
  H = toMPO(ampo)

  # Create an initial random matrix product state
  psi0 = randomMPS(sites)

  # Plan to do 5 passes or 'sweeps' of DMRG,
  # setting maximum MPS internal dimensions
  # for each sweep and maximum truncation cutoff
  # used when adapting internal dimensions:
  sweeps = Sweeps(3)
  maxdim!(sweeps, 2,2,2)
  cutoff!(sweeps, 1E-5)
  @show sweeps

  # Run the DMRG algorithm, returning energy
  # (dominant eigenvalue) and optimized MPS
  energy, psi = dmrg(H,psi0, sweeps)
  println("Final energy = $(energy/N)")
end

  
