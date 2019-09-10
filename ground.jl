using Yao
using BitBasis
using LinearAlgebra
using Statistics
using QuAlgorithmZoo: Sequence
using UnicodePlots

include("circuit.jl")


abstract type AbstractModel{D} end

nspin(model::AbstractModel) = prod(model.size)


struct Heisenberg{D} <: AbstractModel{D}
    size::NTuple{D,Int}
    periodic::Bool
    Heisenberg(size::Int...;periodic::Bool) = new{length(size)}(size,periodic)
end

heisenberg_ij(nbit::Int, m::Int, n::Int = m + 1) = put(nbit,m=>X) * put(nbit,n=>X) + put(nbit,m=>Y) * put(nbit,n=>Y) + put(nbit,m=>Z) * put(nbit,n=>Z)
heisenberg_iijj(nbit::Int, m::Int, n::Int = m + 2) = put(nbit,m=>X) * put(nbit,n=>X) + put(nbit,m=>Y) * put(nbit,n=>Y) + put(nbit,m=>Z) * put(nbit,n=>Z)

function get_bonds(model::AbstractModel{1})
    nbit, = model.size
    [(i,i%nbit+1) for i in 1:(model.periodic ? nbit : nbit-1)]
end

function get_bonds(model::AbstractModel{2})
    m,n = model.size
    cis = LinearIndices(model.size)
    bonds = Tuple{Int,Int,Float64}[]
    for i = 1:m, j = 1:n
        (i!=m || model.periodic) && push!(bonds,(cis[i,j],cis[i%m+1,j]))
        (j!=n || model.periodic) && push!(bonds,(cis[i,j],cis[i,j%n+1]))
    end
    bonds
end

function hamiltionian(model::Heisenberg)
    nbit = nspin(model)
    sum(x->heisenberg_ij(nbit,x[1],x[2]),get_bonds(model))*0.25
end


function gensample(circuit,operator;nbatch=1024)
    mblocks = collect_blocks(Measure,circuit)
    for m in mblocks
        m.operator = operator
    end
    reg = zero_state(nqubits(circuit);nbatch=nbatch)
    reg |> circuit
    mblocks
end

function ising_energy(circuit,bonds,basis;nbatch = nbatch)
    mblocks = gensample(circuit,basis;nbatch = nbatch)
    nspin = length(mblocks)
    local eng = 0.0
    for (a,b) in bonds
        eng += mean(mblocks[a].results.*mblocks[b].results)
    end
    eng/=4
end

function energy(circuit,model::Heisenberg;nbatch = 1024)
    bonds = get_bonds(model)
    sum(basis->ising_energy(circuit,bonds,basis;nbatch = nbatch),[X,Y,Z])
end

function fidelity(circuit,VG)
    psi = zero_state(nqubits(circuit))|>circuit
    return abs(statevec(psi)'*VG)
end

#hei_model = Heisenberg(4;periodic = false)

function cos_fit_min(x::Array{Float64,1},y::Array{Float64,1})
  c = (y[1] + y[3])/2;
  b = atan((y[1] - y[3])/(2*y[2] - y[1] -y[3])) - x[1];
  a = (y[1] - y[3])/(2*sin(x[1] + b));
  return a > 0 ? mod2pi(π + b) : mod2pi(b - π)
end


function train(circuit, model; m = 3, VG = nothing, maxiter = 200, nbatch = 1024)
    rots = Sequence(collect_blocks(RotationGate,circuit))
    loss_history = Float64[]
    for i in 0:maxiter
        params = Float64[]
        for (j,r) in enumerate(rots.blocks)
            E = Float64[]
            tmp = Float64[]
            para = parameters(r)[1]   #parameters return a one-element arrary
            for k in 1:m
                push!(tmp,para)
                push!(E,energy(circuit,model;nbatch=nbatch))
                dispatch!(+,r,(π/2,));
                para += π/2;
            end
            r_op = cos_fit_min(tmp,E);
            dispatch!(r,r_op);
            push!(params,r_op);
        end
        dispatch!(rots,params);
        push!(loss_history,energy(circuit,model,nbatch=nbatch)/nspin(model));

        if i%10==0
            print("Iter $i, E/N = $(loss_history[end])")
            if !(VG == nothing)
                dispatch!(circuit,params)
                fid = fidelity(circuit,VG)
                print(", fidelity = $fid")
                println(",parameters = $params")
            else
                println()
            end
        end
    end
    loss_history,circuit
end

function iscos(mycircuit,model,index = 3,m = 50,nbatch = 1024)
    rots = Sequence(collect_blocks(RotationGate,mycircuit))
    @show rots
    E = Float64[]
    para = Float64[]
    for i in 1:m
        push!(E, energy(mycircuit,model,nbatch = nbatch))
        push!(para,parameters(rots[index])[1])
        dispatch!(rots[index],2.0*π*i/m)
    end
    return E,para
end

#########################################################################
lattice_size = 6;
mycircuit = dispatch!(tcircuit(lattice_size), :random);
mcircuit = dispatch!(fcircuit(lattice_size), :random);
model = Heisenberg(lattice_size;periodic = false)
h = hamiltionian(model)


energy_exact() = expect(h, zero_state(lattice_size) |> mcircuit) |> real
energy_exact()

res = eigen(mat(h)|>Matrix)
EG = res.values[1]/nspin(model)
@show EG
VG = res.vectors[:,1]

E,para = iscos(mycircuit,model,20)
lineplot(para,E)
