#setup and global settings
import lib as pl
import decimal as dc

#understanding single values
my_val = pl.Val("2.3", "1")
print(my_val.v, my_val.e)
print(my_val)

my_other_val = pl.Val(42, "0.1")
print(my_val + my_other_val)
print(my_val * my_other_val)
print(my_val / my_other_val)
print(my_val ** my_other_val)

weighted_mean = pl.Val.weighted_mean([my_val, my_other_val, pl.Val(15, "0.1")])
print(weighted_mean)

lit_value = 42
print(weighted_mean.sigma_interval(lit_value))

#loading data

data = pl.Dataset(csv_path="./sample.csv")
data.auto_bind_errors("delta")

print(data.col("x"))
data.rename_cols([0, "y"], ["time t in s", "distance d in mm"])

def fun(val, row, col):
    return val * pl.Val(1000, 0)


def filter_fun(val):
    return val.e.is_nan() or float(val.e) < 0

data.apply(fun, c_indices=[1])
data.filter(0, filter_fun)
data.title = "my_title"
print(data.to_latex())


print(pl.read_value("./metadata.txt", "n"))

#plotting

data.plot_color = "green"
plot = pl.Plot(point_datasets=[data], curve_datasets=[data], point_column_index_pairs=[[0, 1]], title="My_title")

my_line = pl.Visualizers.Dotted_line(pl.Dataset(lists=[[3, 3], [0, 15000]], c_names=["x", "y"]))
my_text = pl.Visualizers.Text("besondere Linie", [3, 3000], background_color="white")
plot.add_visualizers([my_line, my_text])
plot.title = "irgendwas"
plot.bounds["y"] = [2000, 10000]
plot.x_label = "x"


plot.legend.patches = [pl.Legend_Entry("my_data", "red")]
plot.update_plt()
plot.axes.set_title("hi")
plot.curve_datasets += [data]

plot.show(auto_update=False)

#understanding formulas and variables

foo = pl.Var("\\dot x")

import math as mt

pi = pl.Var("\\pi")

pl.MatEx.constants["my_constant"] = (pi.n, 42)

formula = pl.Formula(variables=[foo], sympy=pi.n*foo.n + 15)
res = formula.at([[foo, pl.Val(3, 1)]])
print("result", res)
print(formula.error.latex)

frml_data = formula.create_values([pl.Val(x, x/10) for x in range(10)])
print(frml_data)
frml_data.rename_cols([0], ["$\dot x$"])
pl.Plot(curve_datasets=[frml_data]).show()

#fitting
my_fit = pl.Fit(data)
print(my_fit.result["m"])

fit_frml = my_fit.formula()
fit_dataset = fit_frml.create_values([pl.Val(x, x/10) for x in range(10)])

pl.Plot(point_datasets=[data], curve_datasets=[fit_dataset]).show()

import sympy as smp

A = pl.Var("A")
x = pl.Var("x")

fit_formula = pl.Formula([x, A], sympy=A.n*smp.sin(x.n))

nl_fit = pl.Fit(data, is_linear=False, fit_formula=fit_formula, x_variable=x, fit_variables=[A])
print(nl_fit.result)

#print(nl_fit.k_fold_cross_validation(10)["CV"])

data.col(1)