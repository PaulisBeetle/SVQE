using Yao;
using LinearAlgebra;
using Statistics:mean
using Random

rng = Random.GLOBAL_RNG;

singletblock(n,i) = chain(n,put(i=>X),put(i=>H),put(i=>X),control(i,i+1=>X),put(i=>X),)
initialblock(n) = chain(n, singletblock(n,i) for i in 1:2:n)
#initialblock(n) = chain(n, i%2==1 ? put(i=>X) : put(i=>Z) for i in 1:n)

A(n,i) = chain(
        n,
        put(i=>Rz(0.0)),
        pswap(n,i,i+1,0.0),
        put(n,i=>Rz(0.0)),
        put(n,i+1=>Rz(0.0)),
        swap(i,i+1),
        )

circuit(n) = chain(n,A(n,i) for i in 1:(n-1));
measure_all(n) = chain(n,Measure{n,1,AbstractBlock,typeof(rng)}(rng,Z,(i,),0,false) for i in 1:n)
tcircuit(n) = chain(n,initialblock(n),circuit(n),measure_all(n))
fcircuit(n) = chain(n,initialblock(n),circuit(n))
