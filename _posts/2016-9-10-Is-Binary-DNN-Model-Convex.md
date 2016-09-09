Is binary DNN model convex?

Another thought that we can take from previous post about our binary DNN model, that it main non-linear element majority function in fact quadratic, and so convex from definitiion(at least at GF(2)).
Put at simple, let's take single activation unit:

    y0 = sign(w0*x0 + w1*x1 + .. + wn*xn)

if we constrain w0..wn and x1..xn to {+1, -1}, than sign function can be replaced with majority:

    y0 = maj(w0*x0, w1*x1, w2*x2, .., wn*xn)

if we take majority with arity 3, than we simple get

    y0 = maj(w0*x0, w1*x1, w2*x2)

where maj is simple:

    x  y  z  maj  
    0  0  0   0
    0  0  1   0
    0  1  0   0
    0  1  1   1
    1  0  0   0
    1  0  1   1
    1  1  0   1
    1  1  1   1

Can we construct maj as quadratic form? Let's try:

	                     [ 0 1 0 ]   [ x ]
    maj(x, y, z) = [x y z] x [ 0 0 1 ] x [ y ] - 2*x*y*z
	                     [ 1 0 0 ]   [ z ]
                         
so it's almost quadratic form, but what is critical, it still *convex*. 
(and what about maj with arbitrary arity? if fact fancy area of algebra exist, named median algebra, that show arbitrary arity can be constructed from 3-arity maj function)
Now we have, for single layer bunch of convex activation function, applied to linear transform of input.
And as it can be easily shown, single layer transformation, is *convex*. But combination(iteration) of convex functions is still convex.
So if we iterate arbitrary number of layers, resulting function is *convex*. Now we just need, to constrain, x1..xn, w1..wn to {-1, +1}, and we get something very similar to convex programming.
As you should see non-linearity(non-convexity) completely gone out from this particular DNN model.
But you can ask how usefull such binary models, there exist research that show(http://arxiv.org/pdf/1511.00363v3.pdf) - this models have similar representation capabilities as tranditional DNN.