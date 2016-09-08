---
layout: post
title: A note about deep learning binary models analysis.
---

A note about deep learning binary models analysis.

There very little known why deep learning models, espessially convolution nets, generalize so well, i'm mean from theorethical stand point. 
The task of this note to develop some simple theorethical toy model for DNN to shead some light on it grounds.
In most real DNN models we operate on vectors/tensors from real line (R^N), and math to analyze full scale model mostly untractable. 
The idea to move from world of real line/manifolds/etc to boolean logic and discret (0,1) state. Some one may thought, that this toy model will have very little connection to reality, but there exist research that use such binary models to train on real datasets and they perform surprisingly well http://arxiv.org/pdf/1511.00363v3.pdf.

A first step to toy model aproximation, activation function for layers, 

	y= sign(x0*w0+x1*w1+..+xn*wn), where x0..xn,w0..wn takes on {-1, +1 }.

At next step we transform activation to boolean algebra, at this stage we use math over GF(2) (mod 2 arithmetic) and n-arity majority function maj, y = maj(x0+w0,x1+w1,x2+w2,..,xn+wn) (changed * to +, because it just xor)
Now we simplify further, and take majority with arity = 3. At this point some simple structure with two deep leveles and stride=1, can be drawn, like:

	maj(w0+maj(x0+w00,x1+w01,x2+w02),w1+maj(x1+w10,x2+w11,x3+w12),w2+maj(x2+w20,x3+w21,x4+w23)), ...

We start with analyzing one simply layer with stride one. At input we take vector X = (x0, x1, x2, .., xn), and produce vector output Y = (y0, y1, y2, .. ym).

	y0=maj(x0+w00,x1+w01,x2+w02)
	y1=maj(x1+w10,x2+w11,x3+w12)
	y2=maj(x2+w20,x3+w21,x4+w23)
	...
	yn

as we know maj(a,b,c) can be transformed to:

	maj(a,b,c) = a&b xor b&c xor a&c

or at GF(2):

	maj(a,b,c) = a*b+b*c+a*c
	y0=maj(x0+w00,x1+w01,x2+w02)=
           w00*w01+w01*w02+w00*w02+
           x0*(w01+w02)+x1*(w00+w02)+x2*(w01+w00)+
           x0*x1+x1*x2+x0*x2
we just can replace w coefficients, with some other new constants c, then:

	y0=x0*c00+x1*c01+x2*c02+x0*x1+x1*x2+x0*x2 or
	y0=[c00 c01 c02]x[x0 x1 x2]^T + x0*x1+x1*x2+x0*x2

 where x0*x1+x1*x2+x0*x2 is the same majority function maj(x0,x1,x2)

	y0 = [c00 c01 c02]x[x0 x1 x2]^T + maj(x0, x1, x2)
	y1 = [c10 c11 c12]x[x1 x2 x3]^T + maj(x1, x2, x3)
	y2 = [c20 c21 c22]x[x2 x3 x4]^T + maj(x2, x3, x4)
	..
	yn 

In general for single layer, we have some matrix(linear operator) acting on input vector X, and non-linear MAJ added.
We have:
	Y_n = L_n(X) + MAJ(X), where L_n, n-level operator with matrix of coefficients, MAJ - majority operator not containing any tunable coefficients.
Or if we take two layer deep:

	Y_1 = L_1(X) + MAJ(X)
	Y_2 = L_2(L_1(X) + MAJ(X)) + MAJ(L_1(X) + MAJ(X)) = L_2(L_1(X)) + L_2(MAJ(X)) + MAJ(L_1(X) + MAJ(X))

interesting non-linear part is:
	MAJ(L_1(X) + MAJ(X))

So what we have do, at each network level up iteration, shuffle input vector by some *linear operation*, apply MAJ operator, that in fact just find correlations at it inputs.
Than take result and put it to the next level, where again simple linear transfomation applied(in GF(2) it some sort of shuffling and negating inputs), and correlations searched, again.
At this point, contrary to believe that DNN models produce exponentially more complex "interpolations", our toy model produced some simple structure, tunabel linear operators on each level applied + fixed structure MAJ non-linear(correlation) operator.