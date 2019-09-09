using Test
include("circuit.jl")
include("ground.jl")

n = 8;

@testset "circuit" begin
    circuit = twoqubit_circuit(n);
    res = gensample(circuit,X; nbatch = 1024)
    hei_model = Heisenberg(4; periodic = false);
    @test energy(circuit,hei_model)
end
