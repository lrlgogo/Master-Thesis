from PFRecAlg import *
from ExampleGenerator import *


def test(prob, u=13):
    fun = PFRecAlgII(u)
    prob.run()
    input = prob.y_image
    fun.get_input(input)
    fun.run()
    fun.show()
    return fun.output


prob_list = [zdt1, zdt2, zdt3, zdt4, zdt6, sch, fon]
"""
for x in prob_list:
    test(x)
"""
out = test(prob_list[1], 15)
