using Yao;
using LinearAlgebra;
using Statistics:mean
using Random

rng = Random.GLOBAL_RNG;

A(n,i) = chain(
        n,
        put(n,i=>Rz(0.0)),
        put(n,i+1=>(i==(n-1) || i > 1 && i%2==0 ? X : Z)),
        put(n,i+1=>Rz(0.0)),
        pswap(n,i,i+1,0.0),
        put(n,i=>Rz(0.0)),
        Measure{n,1,AbstractBlock,typeof(rng)}(rng,Z,(i,),0,false),
        )

circuit(n) = chain(n,A(n,i) for i in 1:(n-1));
twoqubit_circuit(n) = chain(n,circuit(n),put(n,n=>Rz(0.0)),Measure{n,1,AbstractBlock,typeof(rng)}(rng,Z,(n,),0,false))