"""Example for fitting with the physicsLab library"""

from lib import *
import sympy as smp

x = Var("x")
frm = Formula(variables=[x], sympy=smp.sin(x.n))
ds = frm.create_values([i * mt.pi / 5 for i in range(0, 10)], var=x)
ds.add_column([Val("1") for val in ds.col(0)], name="sigma")
ds.bind_error(1, "sigma")
ds.print()

A = Var("A")
phi = Var("phi")
sig_dict = {}

fit = Fit(ds, is_linear=False, fit_formula=Formula([x, A, phi], sympy=A.n*smp.sin(x.n - phi.n)), x_variable=x,
          fit_variables=[A, phi],
          covariance_matrix=Covariance_Matrix(variable_names=["x" + str(i) for i in index_of(ds.col(0))]))

plot = Plot(ds, frm.create_values([i * mt.pi / 500 for i in range(0, 1000)], var=x))
plot.bounds["x"] = [0, 2 * mt.pi]
plot.show()
