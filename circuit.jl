using Yao;
using LinearAlgebra;
using Statistics:mean
using Random

rng = Random.GLOBAL_RNG;

initialblock(n) = chain(n,put(i=>(i==2 ||(i%2==1 && i!=1)) ? X : Z) for i in 1:n)

A(n,i) = chain(
        n,
        put(n,i=>Rz(0.0)),
        put(n,i+1=>Rz(0.0)),
        pswap(n,i,i+1,0.0),
        put(n,i=>Rz(0.0)),
        )

circuit(n) = chain(n,A(n,i) for i in 1:(n-1));
measure_all(n) = chain(n,Measure{n,1,AbstractBlock,typeof(rng)}(rng,Z,(i,),0,false) for i in 1:n)
tcircuit(n) = chain(n,initialblock(n),circuit(n),put(n,n=>Rz(0.0)),measure_all(n))

fcircuit(n) = chain(n,initialblock(n),circuit(n),put(n,n=>Rz(0.0)))
