from lib import *

x = Var("x")
frm = Formula(variables=[x], sympy=smp.sin(x.n))
ds = frm.create_values([i * mt.pi / 5 for i in range(0, 10)], var=x)
ds.print()

A = Var("A")
phi = Var("phi")
sig_dict = {}

fit = Fit(ds, is_linear=False, fit_formula=Formula([x, A, phi], "\\A*sin(x - phi)"), x_variable=x,
          fit_variables=[A, phi],
          covariance_matrix=Covariance_Matrix(variable_names=["x" + str(i) for i in index_of(ds.col(0))]).at(sig_dict))

plot = Plot(ds, frm.create_values([i * mt.pi / 500 for i in range(0, 1000)], var=x))
plot.bounds["x"] = [0, 2 * mt.pi]
plot.show()
