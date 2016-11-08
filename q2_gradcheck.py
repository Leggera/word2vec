import numpy as np
import random
import timeit

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:
        #print x[ix]
        old_xix = x[ix]
        x[ix] = old_xix + h
        random.setstate(rndstate)
        fp = f(x)[0]
        x[ix] = old_xix - h
        random.setstate(rndstate)
        fm = f(x)[0]
        x[ix] = old_xix

        numgrad = (fp - fm)/(2 * h)
        '''
        I = np.zeros_like(x)
        I[ix] = 1
        rndstate = random.getstate()
        fxh_p, _ = f(x + h*I)
        rndstate = random.getstate()
        fxh_m, _ = f(x - h*I)
        numgrad = (fxh_p - fxh_m)/(2*h)'''
        ### END YOUR CODE

        # Compare gradients
        #print grad[ix]
        
        '''print numgrad
        print np.sum(numgrad)
        print grad
        print grad[ix]'''
        '''reldiff = abs(np.sum(numgrad) - grad[ix]) / max(1, abs(np.sum(numgrad)), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print grad
            print fp, fx, fm
            print 2 * h * 10000
            print (fp - fm) * 10000
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return'''
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    
    quad = lambda x: (x.dot(x.T), 2 * x)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    quad = lambda x: (np.sum(x ** 2, axis = 1), x * 2)
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print timeit.timeit('gradcheck_naive(lambda x: (np.sum(x ** 2), x * 2), np.random.randn(4,5))', setup = "import numpy as np; from __main__ import gradcheck_naive", number=1)
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()