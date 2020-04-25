import numpy as np
import theano
import theano.tensor as T

state = theano.shared(np.array(0, dtype=np.float64), 'state')
inc = T.scalar('inc', dtype=state.dtype)

accumulator = theano.function([inc], state, updates=[(state, state+inc)]) # state = state + inc


#print(accumulator(10))
#print(accumulator(10))


print(state.get_value())

accumulator(1)

print(state.get_value())

accumulator(10)

print(state.get_value())

state.set_value(-1)

accumulator(3)

print(state.get_value())

tmp_func = state*2 + inc

a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)]) #replace state by a

print(skip_shared(2,3))

print(state.get_value())
