using Yao
using BitBasis
using LinearAlgebra
using Statistics
using QuAlgorithmZoo: Sequence
using Plots
using LsqFit
using FFTW

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
#   A = std(y);
#    b = mean(y);
#    w = 1.0;
#    ph = 0.0;
#    x0 = vec(x);
#    p = [A,w,ph,b];
#    fun(x0,p) = p[1].*sin.(p[2].*x0 .+ p[3]) .+ p[4];
#    fit = curve_fit(fun,x0,y0,p);
#    return fit.param[1] > 0 ? mod2pi(3/2*π - fit.param[3]) : mod2pi(π/2 - fit.param[3])
    b = (y[1] + y[3])/2
    ϕ = atan((y[1] - b)/(y[2] - b)) - x[1]
    A = (y[1] - b)/sin(x[1] + ϕ)
    if A < 0
        A = -A
        ϕ = mod2pi(ϕ+π)
    else
        ϕ = mod2pi(ϕ)
    end
    return A,ϕ,b
end

function optimalr(A,ϕ,b)
    A = median(A)
    ϕ = median(ϕ)
    b = median(b)
    #A = mean(A)
    #ϕ = mean(ϕ)
    #b = mean(b)
    return mod2pi(3/2*π - ϕ)
end

function train(circuit, model; m = 5, VG = nothing, maxiter = 3, nbatch = 1024)
    rots = Sequence(collect_blocks(RotationGate,circuit))
    mcircuit = fcircuit(nqubits(circuit));
    loss_history = Float64[]
    for i in 0:maxiter
        for (j,r) in enumerate(rots.blocks)
            A = fill(0.0,m)
            ϕ = fill(0.0,m)
            b = fill(0.0,m)
            para = parameters(r)[1]                 #parameters return a one-element arrary
            for k in 1:m                            #m denotes the group of "three points"
                E = Float64[]                       #energy array of three points
                tmp = Float64[]                     #para array of three points
                para += π/2 * (k-1)/m;              #initial para of each group,i.e. the first para
                dispatch!(r,para)                   #parameterized
                para_tmp = copy(para);              #tmp variable
                for l in 1:3
                    push!(tmp,para_tmp)
                    push!(E,energy(circuit,model;nbatch=nbatch))
                    dispatch!(+,r,π/2);
                    para_tmp += π/2;
                end
                A[k],ϕ[k],b[k] = cos_fit_min(tmp,E);
            end
            dispatch!(r,optimalr(A,ϕ,b));
            push!(loss_history,energy(circuit,model,nbatch=nbatch)/nspin(model));
            print("Iter $i.$j, E/N = $(loss_history[end])")
            if !(VG == nothing)
                dispatch!(mcircuit,parameters(circuit))
                fid = fidelity(mcircuit,VG)
                #println(", fidelity = $fid, coefficients of sine function: A, ϕ, b: $A, $ϕ, $b")
                println(", fidelity = $fid")
            else
                println()
            end
        end
        #push!(loss_history,energy(circuit,model,nbatch=nbatch)/nspin(model));
        #if i%10==0
            #print("Iter $i, E/N = $(loss_history[end])")
        #=    if !(VG == nothing)
                dispatch!(mcircuit,parameters(circuit))
                fid = fidelity(mcircuit,VG)
                println(", fidelity = $fid")
            else
                println()
            end=#
        #end
    end
    loss_history,circuit
end

function iscos(mycircuit,model,index = 2,m = 20,nbatch = 1024)
    rots = Sequence(collect_blocks(RotationGate,mycircuit))
    E = Float64[]
    para = Float64[]
    for i in 1:m
        push!(E, energy(mycircuit,model,nbatch = nbatch))
        push!(para,parameters(rots[index])[1])
        dispatch!(+,rots[index],2.0*π/m)
    end
    A = std(E);
    b = mean(E);
    w = 1.0;
    ph = 0.0;
    x0 = vec(para);
    y0 = vec(E);
    p = [A,w,ph,b];
    fun(x0,p) = p[1].*sin.(p[2].*x0 .+ p[3]) .+ p[4];
    fit = curve_fit(fun,x0,y0,p);
    return E,para,fit.param
end


#########################################################################
lattice_size = 6;

model = Heisenberg(lattice_size;periodic = false)
h = hamiltionian(model)
res = eigen(mat(h)|>Matrix)
EG = res.values[1]/nspin(model)
@show EG
VG = res.vectors[:,1]

mycircuit = dispatch!(tcircuit(lattice_size), :random);
npara = nparameters(mycircuit)

loss_history,mycircuit = train(mycircuit,model,m=5;VG = VG)
plot([0:npara*4-1],[loss_history,fill(EG,npara*4)],label = ["QMPS","Exact"],lw = 3,ylabel = "Energy")
E,para,coeffs = iscos(mycircuit,model,15)
plot(para,[E,map(x->coeffs[1].*sin(coeffs[2].*x+coeffs[3])+coeffs[4],para)])
