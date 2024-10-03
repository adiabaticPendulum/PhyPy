import lib as pl

my_val = pl.Val("42", "0.5")
second_val = pl.Val("2.5", "0.1")

print("Operations with", my_val, "and", second_val, ":")
print("\t+:\t", my_val + second_val)
print("\t-:\t", my_val - second_val)
print("\t*:\t", my_val * second_val)
print("\t/:\t", my_val / second_val)
print("\t**:\t", my_val ** second_val)
print("\tabs(my_val):\t", abs(my_val))
print("\tstr(my_val):\t", str(my_val))
print("\tfloat(my_val):\t", float(my_val))
print("\tmy_val.sig_round():\t", my_val.sig_round())

pl.PRINT_OPTIONS["relative_uncertainties"] = True
print("\n\twith relative uncertainties:\t", my_val, "and", second_val)
print("\t\tmy_val.sig_round():\t", my_val.sig_round())

