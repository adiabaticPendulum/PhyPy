# ONLY CHANGE STUFF IN THIS SECTION, NOT IN THE OTHER SECTIONS. SET GLOBAL CONTENTS AND SETTINGS HERE
__debug_lib__ = __debug__  # Weather to show warnings and debug info. Default: __debug__. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application
__debug_extended__ = False  # Weather to show internal debug info (mainly for debugging the library itself).

DEC_DGTS = 64  # How many decimal digits (without rounding errors) shall be used.
#################################################################################################
# Init
ESC_STYLES = {"Error": '\033[41m\033[30m', "Error_txt": '\033[0m\033[31m', "Warning": '\033[43m\033[30m',
              "Warning_txt": '\033[0m\033[33m', "Default": '\033[0m', "Hacker_cliche": "\033[38;2;0;255;0m ", "Green_BG": '\033[42m\033[30m'}

PRINT_OPTIONS = {"relative_uncertainties": False}
"""A dictionary containing flags that alter the output of some functions. Currently, there is only one option: 
If `PRINT_OPTIONS[relative_uncertainties]` is set to `True`, relative uncertainties will be included in the output of some functions that print 'Val' objets, such as `Val.sig_round()`, e.g. "(2 \pm 10) \cdot 10^3 (\pm 20\percent)" instead of just ((2 \pm 10) \cdot 10^3). Default: `False`.
"""

# Note: Appart from this, you have to install 'Jinja2' (e.g. using pip)
import asyncio
import colorsys as cls
import math as mt
import decimal as dc
import copy as cpy
import random as rndm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.patches as ptc
import numpy as np
import pandas as pd
import pandas.io.formats.style as sty
import latex2sympy2 as l2s2
import sympy as smp
import scipy as sp
import sys

libs = [asyncio, cls, mt, plt, np, pd, dc, cpy, l2s2, tck, smp, sp, sys, rndm]

if DEC_DGTS <= dc.MAX_PREC:
    dc.getcontext().prec = DEC_DGTS
else:
    print(ESC_STYLES["Warning"] + "Config Warning: Passed value of" + str(
        DEC_DGTS) + "for DEC_DGTS exceeds allowed precission of " + str(
        dc.MAX_PREC) + " (might vary depending on the system). Using " + str(dc.MAX_PREC) + "instead." + ESC_STYLES[
              "Warning"])

if __debug_lib__:
    print(
    "Running in debug mode. Use 'python <PATH TO YOUR FILE> -oo' to run in optimized mode and hide debug information. Alternatively, go into the physicsLab library source code, find the definition of the variable '__debug_lib__' and turn it to 'False'")
    for lib in libs:
        try:
            print("Using", lib.__name__, lib.__version__)
        except AttributeError:
            pass
    print("\n\n####################################################################\n\n")

# Matplotlib Setup:
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac, amsmath}'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['errorbar.capsize'] = 5
plt.rcParams['lines.markersize'] = 10


########################################################################################################
# internal Functions

def _warn(name, description):
    print("\n" + ESC_STYLES["Warning"] + "WARNING: " + name + ESC_STYLES["Default"] + " " + ESC_STYLES[
        "Warning_txt"] + description + ESC_STYLES["Default"] + "\n")


def _error(name, description):
    print(
        "\n\n\n(" + ESC_STYLES["Error"] + "Error: " + name + ESC_STYLES["Default"] + " " + ESC_STYLES[
            "Error_txt"] + description +
        ESC_STYLES["Default"] + ")\n\n\n")
    raise Exception(name + ": " + description)


######################################################################
# Misc

def index_of(arr, start=0):
    return list(range(start, len(arr)))

def invert_list(list):
    res = [0 for l in list]
    for i in index_of(list):
        res[len(res) - i - 1] = list[i]

    return res

def optimal_indices(arr, target):
    res = []
    epsilon = abs(target-arr[0])
    for i in index_of(arr):
        if abs(target-arr[i]) <= epsilon:
            epsilon = abs(target-arr[i])
    for i in index_of(arr):
        if abs(target-arr[i]) <= epsilon:
            res.append(i)

    return res

def sort_by(list_to_sort, value_function):
    if type(list_to_sort) is not list:
        return list_to_sort
    if sys.getrecursionlimit() < len(list_to_sort):
        sys.setrecursionlimit(len(list_to_sort))
    """sorts list, so that value_function evalueted by the list elements is in ascending order. So instead of sort_by(arr, foo)[i] < sort_by(arr, foo)[i+1] for all 0 <= i < len(arr) - 1, the returned list will forfill foo(sort_by(arr, foo)[i]) < foo(sort_by(arr, foo)[i+1]) for all such i."""
    if len(list_to_sort) == 0:
        return list_to_sort
    pivot = value_function(list_to_sort[0])
    l1 = [val for val in list_to_sort if value_function(val) < pivot]
    l2 = [val for val in list_to_sort if value_function(val) > pivot]
    if len(l1) > 1:
        l1 = sort_by(l1, value_function)
    if len(l2) > 1:
        l2 = sort_by(l2, value_function)
    return l1 + [val for val in list_to_sort if value_function(val) == pivot] + l2


def read_value(path, var_name, seperator="=", use_error=True, error_notation="sigma_", allow_pm=True, pm_notation="\pm", linebreak="\n"):

    inpt = open(path).read().split(var_name)[1].split(seperator)[1].split(linebreak)[0]
    if not use_error:
        return Val(inpt, "NaN")
    if "\pm" in inpt and allow_pm:
        inpt = inpt.split(pm_notation)[0]
        err_inpt = inpt.split(pm_notation)[1]
    else:
        if error_notation + var_name in open(path).read():
            err_inpt = open(path).read().split(error_notation + var_name)[1].split(seperator)[1].split(linebreak)[0]
        else:
            error_notation = "NaN"
    return Val(inpt, err_inpt)

def dictionary_to_file(dictionary, file_path, seperator="=", linebreak="\n", error_notation="sigma_"):
    file = open(file_path, "w")
    for key in dictionary:
        if isinstance(dictionary[key], Val):
            file.write(str(key) + seperator + str(dictionary[key].v) + linebreak)
            file.write(error_notation + str(key) + seperator + str(dictionary[key].e) + linebreak)
        else:
            file.write(str(key) + seperator + str(dictionary[key]))

#####################################################################
# Datasets and Data-Handling

class Solvers:
    class Euler:

        def evaluate(self, resolution, stop=None, variable_values=None):
            res = [[self.initial_condition[i]] for i in index_of(self.initial_condition)]

            client_wants_ds = type(variable_values) is list
            if variable_values is not None:
                stop = variable_values[-1]
                if not client_wants_ds:
                    variable_values = [variable_values]
            else:
                variable_values = []

            variable_values.sort()
            client_values = [[]] + [[] for var in self.result_variables]
            client_values_i = 0
            i = 1
            while True:
                args = {self.variable.n: res[0][i - 1]}
                for k in index_of(self.result_variables):
                    args[self.result_variables[k].n] = res[1 + k][i - 1]
                if len(variable_values) > client_values_i and variable_values[client_values_i] <= res[0][
                    i - 1] + resolution:
                    client_values[0].append(variable_values[client_values_i])
                    for j in index_of(self.result_variables):
                        client_values[1 + j].append(
                            res[1 + j][i - 1] + (variable_values[client_values_i] - res[0][i - 1]) *
                            self.differential_equations[j].sympy.xreplace(args))

                    client_values_i += 1
                res[0].append(res[0][i - 1] + resolution)
                for j in index_of(self.result_variables):
                    res[1 + j].append(
                        res[1 + j][i - 1] + resolution * self.differential_equations[j].sympy.xreplace(args))

                if res[0][i] >= stop:
                    break
                i += 1

            if client_wants_ds:
                return Dataset(c_names=[self.variable.str] + [var.str for var in self.result_variables],
                               lists=client_values)

            if len(client_values) == 1:
                return variable_values[0]

            return Dataset(c_names=[self.variable.str] + [var.str for var in self.result_variables], lists=res)

        def __init__(self, result_variables, variable, differential_equations, initial_condition):
            # tuple initial_condition, MatEx diferential_equation
            self.differential_equations = differential_equations
            self.initial_condition = initial_condition
            self.variable = variable
            self.result_variables = result_variables

    class Root:
        def __init__(self, expression, epsilon):
            self.expression = MatEx(expression.variables.values(), sympy=expression.sympy) if type(expression) is Formula else expression
            delkeys = []
            for key in self.expression.variables.keys():
                if "\\sigma_{" in key:
                    delkeys.append(key)

            for key in delkeys:
                self.expression.variables.__delitem__(key)

            self.epsilon = Val(epsilon)
            self._old_recursion_limit = sys.getrecursionlimit()

        def evaluate(self, initial_guesses):
            if len(initial_guesses) >= self._old_recursion_limit:
                sys.setrecursionlimit(self._old_recursion_limit + len(initial_guesses))
            fun = smp.lambdify([var.n for var in self.expression.variables.values()], self.expression.sympy)
            initial_guess = initial_guesses[0]
            initial_guesses.remove(initial_guess)
            res = sp.optimize.root(fun, initial_guess)

            if len(initial_guesses) > 0:
                ret_ds = self.evaluate(initial_guesses)
            else:
                ret_dict = {}
                for key in self.expression.variables.keys():
                    ret_dict[key] = []
                ret_ds = Dataset(dictionary=ret_dict)

            if not res.success:
                return ret_ds

            for r in index_of(ret_ds.col(0)):
                for c in index_of(ret_ds.row(r)):
                    if not ((ret_ds.at(r, c) - self.epsilon).v <= res.x[c] <= (ret_ds.at(r, c) + self.epsilon).v):
                        break
                    if c == len((ret_ds.row(r))) - 1:
                        # res already in ret_dict
                        return ret_ds

            ret_ds.add_row([Val(component) for component in res.x])
            return ret_ds

        def __del__(self):
            sys.setrecursionlimit(self._old_recursion_limit)

class Val:  # Todo: document!
    """Class to store values with uncertainties."""

    @staticmethod
    def sort_list(val_list, ascending=True):
        for i in index_of(val_list)[1:]:
            if (val_list[i].v > val_list[i-1].v and not ascending) or (val_list[i].v < val_list[i-1].v and ascending):
                tmp = val_list[i-1]
                val_list[i-1] = val_list[i]
                val_list[i] = tmp
                Val.sort_list(val_list)
    @property
    def v(self):
        """The (floating point) value of the Val"""
        return self._v

    @v.setter
    def v(self, new_v):
        """The (floating point) value of the Val"""
        self._v = dc.Decimal(new_v)
        self._known_decimal_figures = 0 if "." not in str(new_v) else len(str(new_v).split('.')[1])
        self._known_decimal_figures += 0 if "e" not in str(new_v) else (-int(str(new_v).split('e')[1]) - 1 - len(str(new_v).split('e')[1]))

    @property
    def e(self):
        """The (floating point) uncertainty of the Val"""
        return self._e

    @e.setter
    def e(self, new_e):
        """The (floating point) uncertainty of the Val"""
        self._e = dc.Decimal(new_e)
        self._e_known_decimal_figures = 0 if "." not in str(new_e) else len(str(new_e).split('.')[1])
        self._e_known_decimal_figures += 0 if "e" not in str(new_e) else (
                    -int(str(new_e).split('e')[1]) - 1 - len(str(new_e).split('e')[1]))

    @staticmethod
    def to_val(val, modify=lambda val: val):
        """Executes a callback function for a variable and returns the result as a Val object. If the callbacks return type can not be cast to Val, the original variable cast to a string will be returned instead.

        Parameters:
            - val: The variable to call modify for.
            - modify: The callback function. Should take only one parameter, which should have the same type as val. Should return a Val object.

        Returns:
            A Val containing the return value of the callback or a string, if the callback has an incompatible return type"""
        if type(val) is Val:
            return modify(val)
        try:
            return Val(modify(val))
        except dc.InvalidOperation:
            return str(val)
        except:
            return Val(float(modify(val)))

    @staticmethod
    def weighted_mean(val_list=None):

        """Calculates the weighted mean of a list of Val objects. If one of the Vals has an invalid error, the (unweighted) mean is returned as a Val object without error.

        Parameters:
            val_list: A list of Val objects. Default: None

        Returns:
            The weighted mean of val_list as a Val object or the (unweighted) mean if one or more members of val_list have an invalid uncertainty, such as 'NaN' or '0'.
        """
        for val in val_list:
            if not isinstance(val, Val):
                return Val(np.mean(val_list))
            if val.e is None or val.e == 0 or mt.isnan(val.e):
                res = Val("0", "0")
                for val2 in val_list:
                    res.v += val2.v
                res.v /= len(val_list)
                return res
        x_over_sig_sq = dc.Decimal(0)
        rez_sig_sq = dc.Decimal(0)

        for val in val_list:
            x_over_sig_sq += dc.Decimal(val.v / (val.e ** 2))
            rez_sig_sq += dc.Decimal(1 / (val.e ** 2))

        return Val(str(x_over_sig_sq / rez_sig_sq), str(mt.sqrt(dc.Decimal(1 / rez_sig_sq))))

    def __init__(self, val, err="NaN"):
        self.v = dc.Decimal(val)
        self.e = dc.Decimal(err)
        self._known_decimal_figures = 0 if "." not in str(val) else len(str(val).split('.')[1])
        self._known_decimal_figures += 0 if "e" not in str(val) else (-int(str(val).split('e')[1]) - 1 - len(str(val).split('e')[1]))
        self._e_known_decimal_figures = 0 if "." not in str(err) else len(str(err).split('.')[1])
        self._e_known_decimal_figures += 0 if "e" not in str(err) else (-int(str(err).split('e')[1]) - 1 - len(str(err).split('e')[1]))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self.v == other.v and self.e == other.e

    def __str__(self):
        return self.sig_round(warn_for_bad_error=False)[0]
        # return str(self.v) if mt.isnan(self.e) or type(self.e) == str else self.sig_round()[0]

    def __float__(self):
        return float(self.v)

    def __truediv__(self, other):
        x = Var("x")
        y = Var("y")
        return Formula([x, y], "x/y").at([[x, self], [y, other]], as_val=True)

    def __add__(self, other):
        x = Var("x")
        y = Var("y")
        return Formula([x, y], "x+y").at([[x, self], [y, other]], as_val=True)

    def __sub__(self, other):
        x = Var("x")
        y = Var("y")
        return Formula([x, y], "x-y").at([[x, self], [y, other]], as_val=True)

    def __mul__(self, other):
        x = Var("x")
        y = Var("y")
        return Formula([x, y], "x*y").at([[x, self], [y, other]], as_val=True)

    def __pow__(self, other):
        x = Var("x")
        y = Var("y")
        return Formula([x, y], sympy=x.n**y.n).at([[x, self], [y, other]], as_val=True)

    def __abs__(self):
        x = cpy.deepcopy(self)
        x.v = abs(x.v)
        return x

    def get(self):
        return cpy.deepcopy(self.v)

    def set(self, val):
        self.v = val
        self._known_decimal_figures = 0 if "." not in str(val) else len(str(val).split('.')[1])
        self._known_decimal_figures += 0 if "e" not in str(val) else (
                    -int(str(val).split('e')[1]) - 1 - len(str(val).split('e')[1]))

    def get_err(self):
        return cpy.deepcopy(self.e)

    def set_err(self, new_err):
        self.e = new_err
        self._e_known_decimal_figures = 0 if "." not in str(new_err) else len(str(new_err).split('.')[1])
        self._e_known_decimal_figures += 0 if "e" not in str(new_err) else (
                    -int(str(new_err).split('e')[1]) - 1 - len(str(new_err).split('e')[1]))

    def sig_round(self, sig_digits=1, additional_digit=True, warn_for_bad_error=True):
        """Rounds the Val to a given number of significant digits.

        Parameters:
            sig_digits: The number of significant digits to round to. Should be of type int. Default: 1
            additional_digit: Weather to round to an additional digit if the first additional digit is a 1 or a 2. Default: True
            warn_for_bad_error: Weather to create a warning the Val has an invalid error, such as "NaN" or 0.

        Returns:
        A list with the following members (in that order):
        - A string describing the value with the corresponding uncertainty in the format ([value] \pm [uncertainty]) \cdot 10^{[decimal exponent]}. If PRINT_OPTIONS["relative_uncertainties"] is True, the relative uncertainty will be printed as well: ([value] \pm [uncertainty]) \cdot 10^{[decimal exponent]} \: (\pm [relative uncertainty] \\%)
        - A string describing the value
        - A string describing the (absolute) uncertainty
        - A string describing the relative uncertainty (in percent)
        """
        val = self.get()
        err = self.get_err()
        if mt.isnan(err) or err <= 0:
            if warn_for_bad_error:
                _warn("ValueWarning",
                      "Can't sig_round() Val with error 'NaN' or '0', but got (" + str(val) + ", " + str(err) + ")")
            if val == 0:
                return ["0"]
            if val.is_nan():
                return ["NaN"]

            dec_pot = dc.Decimal(mt.floor(mt.log10(abs(val))) - sig_digits - 1)
            val *= dc.Decimal(10 ** -dec_pot)
            val = round(val)
            return [str(val * dc.Decimal(10 ** dec_pot)) if abs(dec_pot) < 3 else str(val) + " \cdot 10^{" + str(
                dec_pot) + "}"]

        if mt.isnan(err) or type(val) is str or type(err) is str:
            _error("Invalid Value",
                   "Can only sig_round Vals with well-defined values and error, but got " + str(val) + ", " + str(err))
            return

        if val.is_nan():
            return ["NaN"]
        sig_pos = mt.floor(mt.log10(
            err))  # position of the first significant figure relative to the decimal point (Number of digits between first significant digit and point, can be negative (behind point), 0 is in front of point)

        if __debug_extended__:
            print("Position of first significant digit: ", sig_pos)

        dec_pot = -(
            -sig_pos + sig_digits - 1 if -sig_pos + sig_digits - 1 <= self._e_known_decimal_figures else self._e_known_decimal_figures)  # digits the point gets shifted by
        err *= dc.Decimal(10 ** -dec_pot)
        if additional_digit and err < 3:
            err *= dc.Decimal(10)
            dec_pot -= 1

        err = mt.ceil(err)
        dec_string = ("" if dec_pot == 0 else " \cdot 10^{" + str(dec_pot) + "}")
        str_err = str(err) + dec_string

        val *= dc.Decimal(10 ** -dec_pot)
        val = round(val)
        str_val = str(val) + dec_string

        if self.v == 0:
            percent = "NaN"
        else:
            percent = abs(dc.Decimal(100) * self.e / self.v)
            perc_pot = mt.floor(mt.log10(percent)) - 1
            if abs(perc_pot + 1) <= 3:
                if perc_pot >= 0:
                    percent = mt.ceil(percent)
                else:
                    percent *= dc.Decimal(10) ** dc.Decimal(-perc_pot)
                    percent = mt.ceil(percent)
                    percent *= dc.Decimal(10) ** dc.Decimal(perc_pot)
                percent = str(percent)
            else:
                percent *= dc.Decimal(10) ** dc.Decimal(-perc_pot)
                percent = mt.ceil(percent)
                percent = str(percent) + " \cdot 10^{" + str(perc_pot) + "}"

        ret = "(" + str(val) + " \\pm " + str(err) + ")" + dec_string
        return [ret + " \: (\\pm " + percent + "\\%)" if PRINT_OPTIONS["relative_uncertainties"] else ret, str_val, str_err, percent]

    def sigma_interval(self, true_value):
        """Returns the smallest sigma environment around a true value the `Val` is in.

        **Parameters**:
        - `true_value`: A float describing the true value.

        **Returns**:
        A float describing the offset of the true value and the `Val`s value as a fraction of the `Val`s uncertainty. E.g. `Val("1", "0.5").sigma_interval(3.25)` will yield `4.5` as $`\left|\frac{3.25 - 1}{0.5}\right| = 4.5`$.
        """
        if type(true_value) is Val:
            true_value = true_value.v
        if type(true_value) is not dc.Decimal:
            true_value = dc.Decimal(true_value)

        return abs(true_value - self.v)/self.e

class Var:
    n = ""
    str = ""

    def __init__(self, name):
        self.str = name
        self.n = smp.Symbol(name)

    def __str__(self):
        return self.str

    def __float__(self):
        return self.n

class MatEx:

    constants = {"PI": (smp.Symbol("PI"), smp.pi), "EULER": (smp.Symbol("EULER"), smp.euler)}

    @staticmethod
    def define_constant(name, value, key=None):
        if type(name) is str:
            name = smp.Symbol(name)

        if key is None:
            key = str(name)
        MatEx.constants[key] = (name, value)

    @property
    def sympy(self):
        return self._sympy

    @sympy.setter
    def sympy(self, new_sympy):
        self._raw_sympy = new_sympy
        self._sympy = self._raw_sympy.doit()
        self._latex = smp.latex(self._sympy)

    @property
    def latex(self):
        return "$" + self._latex + "$"

    @latex.setter
    def latex(self, new_latex):
        self.sympy = l2s2.latex2sympy(new_latex)

    def __init__(self, variables, latex="", sympy=None):
        self.variables = {}
        for var in variables:
            if type(var) is str:
                self.variables[var] = Var(var)
            else:
                self.variables[var.str] = var
        self.sympy = sympy if sympy is not None else (None if latex == "" else l2s2.latex2sympy(latex))
    def __str__(self):
        return self.latex

    def at(self, var_val_pairs):
        tmp_sympy = cpy.deepcopy(self._sympy).doit()
        for var_val_pair in var_val_pairs:
            if isinstance(var_val_pair[0], Var):
                var_val_pair[0] = var_val_pair[0].n

            tmp_sympy = tmp_sympy.subs(var_val_pair[0], var_val_pair[1])

        tmp_sympy = self._substitute_constants(tmp_sympy)
        return tmp_sympy

    def clone(self):
        return cpy.deepcopy(self)

    def local_extrema(self, epsilon, x_variable=None, include_maxima=True, include_minima=True):
        """epsilon describes the resolutoíon of the extrema: if the lowest difference between two minimal points is less than epsilon, they are considered the same and one is left out in the dataset to be returned"""
        if x_variable is None:
            x_variable = list(self.variables.values())[0]
        if type(x_variable is not Var):
            try:
                x_variable = self.variables[x_variable]
            except KeyError:
                _error("ValueError", "Invalid x_varibale:" + str(x_variable) + "\nThere is no such variable in Expression:" + self.latex)
                return -1
        derivative = self.clone()
        derivative.sympy = smp.Derivative(self.sympy, x_variable.n)
        slv = Solvers.Root(derivative, epsilon)
        return slv

    @staticmethod
    def _substitute_constants(sympy):
        initial_sympy = cpy.deepcopy(sympy)
        for key in MatEx.constants.keys():
            sympy = sympy.subs(MatEx.constants[key][0], MatEx.constants[key][1])

        if sympy != initial_sympy:
            sympy = MatEx._substitute_constants(sympy)
        return sympy

class Formula(MatEx):
    @staticmethod
    def from_mat_ex(mat_ex):
        return Formula(variables=list(mat_ex.variables.keys()), sympy=mat_ex.sympy)

    def update_errors(self):
        _err = 0
        for var in list(self.variables.keys()):
            if "\\sigma_{" not in var and "\\sigma_{" + var + "}" not in list(self.variables.keys()):
                self.variables["\\sigma_{" + var + "}"] = Var('\\sigma_{' + var + '}')
        for key in self.variables.keys():
            if "\\sigma_{" not in key:
                _err += (smp.Derivative(self.sympy, self.variables[key].n).doit(deep=False) * self.variables[
                    "\\sigma_{" + str(key)+"}"].n) ** 2
        self.error = MatEx(variables=list(list(self.variables.values())), sympy=smp.sympify(smp.sqrt(_err)))

        if __debug_extended__:
            print("Updated errors of " + self.latex + " to " + self.error.latex)

    @property
    def error(self):
        err = cpy.deepcopy(self._error)
        for key in self.preset_variables.keys():
            err.sympy = err.sympy.subs(self.variables[key].n, self.preset_variables[key])
        return err

    @error.setter
    def error(self, new_error):
        self._error = new_error

    @property
    def sympy(self):
        # in order to take into account preset variables
        preset_sympy = self._sympy
        for key in self.preset_variables.keys():
            preset_sympy = preset_sympy.subs(self.variables[key].n, self.preset_variables[key])
        return preset_sympy

    @sympy.setter
    def sympy(self, new_sympy):
        MatEx.sympy.fset(self, new_sympy)
        self.update_errors()

    @property
    def latex(self):
        # in order to take into account preset variables
        frml = self.clone()
        for key in self.preset_variables.keys():
            frml.sympy = frml.sympy.subs(self.variables[key].n, self.preset_variables[key].v)

        for atom in frml.sympy.atoms(smp.Float):
            frml.sympy = frml.sympy.subs(atom, smp.N(atom, 2))

        frml.sympy = smp.sympify(frml.sympy)
        try:
            frml.sympy = smp.simplify(frml.sympy)
        except:
            pass

        return "$" + frml._latex + "$"

    @latex.setter
    def latex(self, new_latex):
        MatEx.latex.fset(self, new_latex)
        self.update_errors()

    def __init__(self, variables=None, latex="", sympy=None):
        self.preset_variables = {}
        self.error = None
        if variables is None:
            variables = []
        MatEx.__init__(self, variables, latex, sympy)

    def __str__(self):
        ret_err = self.error.sympy
        ret_err = ret_err.doit()
        ret_err = smp.sympify(ret_err)
        ret_err = smp.simplify(ret_err)
        for atom in ret_err.atoms(smp.Float):
            ret_err = ret_err.subs(atom, smp.N(atom, 2))

        return self.latex[1:-1] + " \\pm " + smp.latex(ret_err)

    def at(self, var_val_pairs, as_val=True):
        for variable in self.preset_variables.keys():
            var_val_pairs.append([self.variables[variable], self.preset_variables[variable]])

        for i in index_of(var_val_pairs):
            if isinstance(var_val_pairs[i][1], Val) and "\\sigma_{" not in var_val_pairs[i][0].str:
                var_val_pairs.append([self.variables["\\sigma_{" + var_val_pairs[i][0].str + "}"], var_val_pairs[i][1].e])
                var_val_pairs[i][1] = var_val_pairs[i][1].v

        if as_val:
            val = MatEx.at(self, var_val_pairs).evalf()
            err = self.error.at(var_val_pairs).evalf()
            return Val(str(val), str(err))
        if self.error is None:
            return MatEx.at(self, var_val_pairs), None
        return MatEx.at(self, var_val_pairs), self.error.at(var_val_pairs)

    def set_variables(self, var_val_pairs):
        for var_val_pair in var_val_pairs:
            val = cpy.deepcopy(var_val_pair[1])
            val.e = dc.Decimal("NaN")
            self.preset_variables[var_val_pair[0].str] = val
            self.preset_variables["\\sigma_{" + var_val_pair[0].str + "}"] = Val(var_val_pair[1].e)

    def substitute_variables(self, var_formula_pairs):
        for pair in var_formula_pairs:
            self.sympy = self.sympy.subs(pair[0].n, pair[1].sympy)
            self.error.sympy = self.error.sympy.subs(self.error.variables["\\sigma_{" + pair[0].str + "}"].n, pair[1].error.sympy)
            for dict_entry in pair[1].variables.items():
                self.variables[dict_entry[0]] = dict_entry[1]
            for dict_entry in pair[1].error.variables.items():
                self.error.variables[dict_entry[0]] = dict_entry[1]

            del self.variables[pair[0].str]
            del self.error.variables[pair[0].str]
            del self.error.variables["\\sigma_{" + pair[0].str + "}"]
            del self.variables["\\sigma_{" + pair[0].str + "}"]


    def to_val(self, var_val_pairs):
        at = self.at(var_val_pairs, as_val=False)
        try:
            return Val(str(at[0].evalf()), str(at[1].evalf()))
        except dc.InvalidOperation:
            _error(name="ConversionError",
                   description="Can't convert sympy (" + str(at[0].evalf()) + "\\pm" + str(at[
                                                                                               1].evalf()) + ") to Val. Make shure that the specified key-value-pairs are providing values for all used variables, so that the sympy-expressioon can be evaluated to a numeric expression and doesn't contain any unset variables.")

    def clone(self):
        return cpy.deepcopy(self)

    def create_values(self, var_values, var=None, val_label=None):
        if val_label is None:
            val_label = self.latex
        if var is None:
            var = list(self.variables.values())[0].str
        if type(var) is Var:
            var = var.str

        err_label = "\sigma_{" + val_label + "}"
        data = {var: var_values, val_label: []}
        preset_sympy = self.sympy.subs(self.variables["\\sigma_{" + var + "}"].n, 0)
        preset_err_sympy = self.error.sympy.subs(self.variables["\\sigma_{" + var + "}"].n, 0)
        for key in self.preset_variables.keys():
            preset_err_sympy.subs(self.variables[key].n, self.preset_variables[key])

        preset_sympy = self._substitute_constants(preset_sympy)
        preset_err_sympy = self._substitute_constants(preset_sympy)

        fast_val = smp.utilities.lambdify(self.variables[var].n, preset_sympy)
        fast_err = smp.utilities.lambdify(self.variables[var].n, preset_err_sympy)
        for var_val in var_values:
            if isinstance(var_val, Val):
                var_val = float(var_val.v)
            val = Val(fast_val(var_val), str(fast_err(var_val)))
            if val.e == 0:
                val.e = dc.Decimal("NaN")
            data[val_label].append(val)

        return Dataset(dictionary=data)  # , r_names=[var, val_label]

class Dataset:  # Object representing a full Dataset

    def __str__(self):
        max_rows = pd.get_option('display.max_rows')
        max_cols = pd.get_option('display.max_columns')
        max_col_width = pd.get_option('display.max_colwidth')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        res = str(self.frame)
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_cols)
        pd.set_option('display.max_colwidth', max_col_width)
        return res
    def __add__(self, other):
        cpy = self.clone()
        cpy.join(other)
        return cpy

    def __init__(self, x_label=None, y_label=None, dictionary=None, lists=None, csv_path=None, r_names=None,
                 c_names=None, val_err_index_pairs=None, title=None):
        self.frame = pd.DataFrame()
        self.title = title
        self.plot_color = None
        if val_err_index_pairs is None:
            val_err_index_pairs = []
        if dictionary is not None:
            self.from_dictionary(dictionary, r_names)
        elif lists is not None:
            self.from_lists(lists, r_names, c_names)
        elif csv_path is not None:
            self.from_csv(csv_path)

        self.bind_errors(val_err_index_pairs)

        self.x_label = x_label
        self.y_label = y_label
        self.frame.index = self.frame.index.astype(str, False)

    def row(self, index):
        """Returns the row at the given index.

        **Parameters**:
        - `index`: Which row to return The index can be either the integer index of the row or the row name.

        **Returns**:
        The row contents, cast to a list."""
        try:
            res = self.frame.loc[index]
        except:
            res = self.frame.iloc[index]
        return list(res)

    def rename_rows(self, indices, new_names):
        name_dict = {}
        for i in index_of(indices):
            name_dict[self.get_row_names()[indices[i]]] = new_names[i]
        self.frame.rename(mapper=name_dict, inplace=True, axis="rows")

    def rename_cols(self, indices, new_names):
        name_dict = {}
        for i in index_of(indices):
            #name_dict[self.get_col_names()[indices[i]]] = new_names[i]
            name_dict[indices[i] if type(indices[i]) is str else self.get_col_names()[indices[i]]] = new_names[i]

        self.frame.rename(mapper=name_dict, inplace=True, axis="columns")

    def add_column(self, content, name, index=None):
        if index is None:
            index = len(self.get_col_names())
        self.frame.insert(loc=index, column=name, value=content)

    def add_row(self, content):
        self.frame = pd.concat([self.frame, Dataset(lists=[[c] for c in content], c_names=self.get_col_names()).frame],
                               axis="index", ignore_index=True)
        self.frame.index = self.frame.index.astype(str, False)

    def col(self, index):
        if type(index) is not int:
            res = self.frame[index]
        else:
            res = self.frame[self.get_col_names()[index]]
        return list(res)

    def at(self, r_index, c_index):
        r_name = r_index if type(r_index) is not int else self.get_row_names()[r_index]
        c_name = c_index if type(c_index) is not int else self.get_col_names()[c_index]
        return self.frame.at[r_name, c_name]

    def set(self, r_index, c_index, value):
        r_name = r_index if type(r_index) is not int else self.get_names([r_index, c_index])[0]
        c_name = c_index if type(c_index) is not int else self.get_names([r_index, c_index])[1]
        self.frame.at[r_name, c_name] = value

    def disp_row(self, index):
        print(self.row([index]))

    def disp_col(self, index):
        print(self.col([index]))

    def get_names(self, location):
        for i in index_of(location):
            if type(location[i]) is not int:
                location[i] = list(self.frame.columns).index(location[i])

        return [self.frame.index[location[0]],
                self.frame.columns[location[1]]]

    def get_row_names(self):
        return self.frame.index.to_list()

    def get_col_names(self):
        return self.frame.columns.to_list()

    def apply(self, method, r_indices=None, c_indices=None):
        if r_indices is None:
            r_indices = index_of(self.get_row_names())
        if c_indices is None:
            c_indices = self.get_col_names()

        if type(r_indices) is not list:
            r_indices = [r_indices]
        if type(c_indices) is not list:
            c_indices = [c_indices]

        for r_index in r_indices:
            for c_index in c_indices:
                self.set(r_index, c_index, method(self.at(r_index, c_index), r_index, c_index))

    def print(self, extended=False):
        if extended:
            max_rows = pd.get_option('display.max_rows')
            max_cols = pd.get_option('display.max_columns')
            max_col_width = pd.get_option('display.max_colwidth')
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', None)
        print(self.frame)
        if extended:
            pd.set_option('display.max_rows', max_rows)
            pd.set_option('display.max_columns', max_cols)
            pd.set_option('display.max_colwidth', max_col_width)

    def delete(self, c_indices=None, r_indices=None):
        """deletes the rows described by row indices in the provided list r_indices. In the same way, this deletes the columns described by the column indices found in the provided list c_indices"""
        if c_indices is None:
            c_indices = []

        if r_indices is None:
            r_indices = []

        if not type(r_indices) is list:
            r_indices = [r_indices]

        if not type(c_indices) is list:
            c_indices = [c_indices]

        if len(c_indices) == 0 and len(r_indices) == 0:
            return

        c_names = []
        r_names = []

        if type(c_indices) is not list:
            c_names.append(c_indices if type(c_indices) is not int else self.get_col_names()[c_indices])
        else:
            for c_index in c_indices:
                c_names.append(c_index if type(c_index) is not int else self.get_col_names()[c_index])

        if type(r_indices) is not list:
            r_names.append(r_indices if type(r_indices) is not int else self.get_row_names()[r_indices])
        else:
            for r_index in r_indices:
                r_names.append(r_index if type(r_index) is not int else self.get_row_names()[r_index])

        r_names = sort_by(r_names, lambda val: len(self.get_row_names()) - self.get_row_names().index(val))
        c_names = sort_by(c_names, lambda val: len(self.get_col_names()) - self.get_col_names().index(val))

        for r_name in r_names:
            self.frame.drop(index=r_name, inplace=True)
        for c_name in c_names:
            self.frame.drop(columns=c_name, inplace=True)

    def bind_error(self, value_col_index, error_col_index):
        val_col = self.col(value_col_index)
        err_col = self.col(error_col_index)

        for i in index_of(err_col):
            if isinstance(val_col[i], Val):
                val_col[i].e = err_col[i].v
        self.delete(c_indices=error_col_index)

    def bind_errors(self, val_err_index_pairs):
        for pair in val_err_index_pairs:
            self.bind_error(pair[0], pair[1])

    def c_index_to_c_name(self, c_index):
        if type(c_index) is int:
            c_index = self.get_col_names()[c_index]
        return c_index

    def c_index_to_number(self, c_index):
        if c_index in self.get_col_names():
            c_index = self.get_col_names().index(c_index)
        return c_index

    def unbind_errors(self, c_indices):
        for c_index in c_indices:
            self.add_column([entry.e if isinstance(entry, Val) else None for entry in self.col(c_index)], "sigma_{" + str(self.c_index_to_c_name(c_index)) + "}")

        self.apply(lambda val, r, c: val.v if isinstance(val, Val) else val, c_indices=c_indices)

    def from_dictionary(self, dictionary, r_names=None,
                        items=None):  # Initialize the Dataset with a Python dictionary. Parameters: dictionary: The dictionary of lists to read the data from. items (otional): Which items (lists) of the dictionary to use. Default is 'None', which will use all items.
        data = {}
        if items == None:
            items = list(dictionary.keys())
        if __debug_extended__:
            print("Using items", items, "for creation")

        if not isinstance(dictionary, dict):
            _error("Type Error",
                   "Parameter 'dictionary' must be of Type <class 'dict'>, but has type " + str(type(dictionary)))
            return -1

        for item in items:
            data[item] = []
            for val in dictionary[item]:
                if not type(val) is Val:
                    data[item].append(Val.to_val(val))
                else:
                    data[item].append(val)

        if r_names is not None:
            self.frame = pd.DataFrame(data, r_names)
        else:
            self.frame = pd.DataFrame(data)

        self.frame.index = self.frame.index.astype(str, False)

    def from_lists(self, lists, r_names=None, c_names=None, strict=False):
        if c_names is None:
            c_names = []

        if r_names is None:
            r_names = range(len(lists[0]))

        if len(lists) != len(c_names):
            if strict:
                _error("Parameter Length Missmatch", "Parameters 'lists' and 'c_names' must have the same length but "
                                                     "have shapes (" + str(len(lists)) + ") and ("
                       + str(len(c_names)) + ").")
                return -1
            if __debug_lib__:
                _warn("Parameter Length Missmatch", "Parameters 'lists' and 'c_names' must have the same length but "
                                                    "have shapes (" + str(len(lists)) + ") and (" + str(len(c_names)) +
                      "). Missing c_names will be initialized with standard indices, missing lists with all 'None'.")
            while len(c_names) < len(lists):
                c_names.append(len(c_names))
            while len(lists) < len(c_names):
                lists.append([None for i in lists[0]])
        try:
            data = {}
            for i in index_of(c_names):
                data[c_names[i]] = lists[i]
            self.frame = pd.DataFrame(data)
        except ValueError:
            max_len = len(lists[0])
            for i in index_of(lists, start=1):
                if len(lists[i]) != len(lists[i - 1]):
                    if strict:
                        _error("ValueError",
                               "All items in 'lists' must have same dimension, but items at Indices " + str(
                                   i - 1) + " and " + str(i) + " have shapes (" + str(
                                   len(lists[i - 1])) + ") and (" + str(len(lists[i])) + ").")
                        return -1
                    _warn("ValueWarning", "All items in 'lists' must have same dimension, but items at Indices " + str(
                        i - 1) + " and " + str(i) + " have shapes (" + str(len(lists[i - 1])) + ") and (" + str(
                        len(lists[i])) + "). Short items will be filled with 'NaN' of type dc.Decimal")
                    max_len = np.max([len(lists[i]), len(lists[i - 1]), max_len])
                    if __debug_extended__:
                        print("Maximal detected list-length:", max_len)
            for i in index_of(lists):
                while len(lists[i]) < max_len:
                    lists[i].append(dc.Decimal('NaN'))
        data = {}
        for i in index_of(c_names):
            data[c_names[i]] = []
            for j in index_of(lists[i]):
                try:
                    data[c_names[i]].append(Val.to_val(lists[i][j]))
                except TypeError:
                    data[c_names[i]].append(lists[i][j])

        self.frame = pd.DataFrame(data, r_names)
        self.frame.index = self.frame.index.astype(str, False)

    def from_csv(self, path, delimiter=None, c_names_from_row=0, c_names=None, indices_from_row=None, usecols=None,
                 userows=None, NaN_alias="NaN", compression=None, strict=False, modify_cols={}, modify_rows={}):
        # TODO: TEST

        temp = pd.read_csv(filepath_or_buffer=path, sep=delimiter, header=c_names_from_row, names=c_names,
                           index_col=indices_from_row, na_values=NaN_alias, na_filter=True,
                           verbose=__debug_extended__, compression=compression, quotechar="\"", comment="#",
                           on_bad_lines='error' if strict else 'warn', dtype=object)

        shp = temp.shape
        for r in range(shp[0]):
            for c in range(shp[1]):
                temp.iloc[r].iloc[c] = Val.to_val(temp.iloc[r].iloc[c])

        for k in modify_cols.keys():
            for i in index_of(temp.get[k]):
                temp.get[k][i] = modify_rows[k](temp.get[k][i])

        for k in modify_rows.keys():
            for i in index_of(temp.loc[k]):
                temp.loc[k][i] = modify_rows[k](temp.loc[k][i])

        if userows is None:
            userows = range(shp[0])
        if usecols is None:
            usecols = range(shp[1])

        temp = temp.iloc[:, [col for col in usecols]]
        self.frame = temp.loc[userows]
        self.apply(lambda obj, r_index, c_index: Val.to_val(obj))
        self.frame.index = self.frame.index.astype(str, False)

    def filter(self, c_index, filter_method):
        """deletes every row from the dataset, for which the provided function filter_method returns True when called with the value in that row and the column described by c_index"""
        if type(c_index) is str:
            c_index = self.get_col_names().index(c_index)
        #if c_index <= len(self.col(c_index)):
            #return
        col = list(self.col(c_index))
        for i in index_of(col):
            if filter_method(col[i]):
                self.delete(r_indices=i)
                self.filter(c_index, filter_method)
                break

    def to_csv(self, path, delimiter=";", columns=None, show_index=False, exact=True):
        if exact:
            for c_name in self.get_col_names():
                self.add_column([(val.e if not mt.isnan(val.e) else "NaN") if isinstance(val, Val) else "NaN" for val in self.col(c_name)], "sigma_{" + c_name + "}")
            self.apply(lambda val, r, c: val.v if isinstance(val, Val) else val)
        if columns is None:
            columns = self.get_col_names()
        self.frame.to_csv(path, sep=delimiter, index_label=self.x_label, columns=columns, index=show_index)

    def auto_bind_errors(self, error_notaition = "sigma_{"):
        for c_name in self.get_col_names():
            for c2_name in self.get_col_names():
                if error_notaition == "sigma_{":
                    if "sigma_{" + c_name + "}" in c2_name or "sigma_" + c_name in c2_name or ("$" in c_name and ("sigma_{" + c_name.split("$")[1] + "}" in c2_name or "sigma_" + c_name.split("$")[1] in c2_name)):
                        self.bind_error(c_name, c2_name)
                else:
                    if error_notaition + c_name in c2_name or "$" in c_name and error_notaition + (error_notaition+c2_name.split("$")[1]) in c2_name:
                        self.bind_error(c_name, c2_name)

    def to_latex(self, show_index=False, hline_all=False):

        ds = cpy.deepcopy(self)
        ds.apply(lambda cell, row, col: "$" + str(cell) + "$" if isinstance(cell, Val) else cell)
        styler = sty.Styler(ds.frame)
        if not show_index:
            styler.hide(axis="index")

        col_frmt = "|l|"
        for col in self.get_col_names():
            col_frmt += "l|"

        ltx = styler.to_latex(position_float="centering", label="tab:my_table", caption="\\todo{caption}",
                              column_format=col_frmt)
        ltx = ltx.replace("\\begin{tabular}{" + col_frmt + "}", "\\begin{tabular}{" + col_frmt + "}" + "\n\hline")

        if hline_all:
            ltx = ltx.replace("\\\\", "\\\\ \hline")
            ltx = ltx.replace("\\\\ \hline", "\\\\ \\specialrule{0.08em}{0em}{0em}", 1)
        else:
            ltx = ltx.replace("\\\\", "\\\\ \hline", 1)
            ltx = ltx.replace("\end{tabular}", "\hline\n\end{tabular}")
        return ltx

    def clone(self):
        res = cpy.deepcopy(self)
        res.apply(lambda val, x, y: cpy.deepcopy(val))
        return res

    def move_row(self, old_index, new_index):
        tmp_rows = [self.row(old_index)]

        if type(new_index) is not int:
            new_index = self.get_row_names().index(new_index)
        if type(old_index) is not int:
            old_index = self.get_row_names().index(old_index)

        row_names_to_shift = self.get_row_names()[new_index:]
        for i in row_names_to_shift:
            tmp_rows.append(self.row(i))

        self.delete(r_indices=row_names_to_shift)
        if not self.get_row_names()[old_index] in row_names_to_shift:
            self.delete(r_indices=[old_index])

        for tmp_row in tmp_rows:
            self.add_row(tmp_row)

    def sort(self, column_index, ascending=True):
        col = self.col(column_index)
        indices = index_of(col)
        indices = sort_by(indices, lambda val: col[val]._v)
        new_ds = self.clone()
        new_ds.delete(r_indices=self.get_row_names())
        if not ascending:
            invert_list(indices)
        for i in indices:
            new_ds.add_row(self.row(i))
        self.frame = new_ds.frame

    def local_extrema(self, y_index=1, x_index=0, include_maxima=True, include_minima=True, smoothing_radius=1, difference_radius=None, minimal_absolute_difference=0, minimal_relative_difference=0, minimal_difference_relative_to_biggest_absolute_extremum=0):
        """smoothing radius defines, with how many neighboring values each potential extremum point is compared. minimal_absolute_difference and minimal_relative_difference are the required difference, a extremum has to have to another value in its difference_radius (absolute and relative to the extremums value)"""
        if type(x_index) is not int:
            x_index = self.get_col_names()[x_index]
        if type(y_index) is not int:
            y_index = self.get_col_names()[y_index]
        if difference_radius is None:
            difference_radius = len(self.col(y_index))
        res = self.clone()
        res.filter(y_index, lambda val: type(val) is not Val)
        res.filter(x_index, lambda val: type(val) is not Val)

        self.sort(x_index)
        y_col = res.col(y_index)
        delete_indices = []
        biggest_absolute_extremum = max([abs(val.v) for val in y_col])
        for i in index_of(y_col):
            greatest_diff = 0
            for k in range(1, smoothing_radius):
                if include_minima and (i-k < 0 or y_col[i-k].v >= y_col[i].v) and (i+k > len(y_col[1:-1]) or y_col[i].v <= y_col[i+k].v):
                    continue
                if include_maxima and (i-k < 0 or y_col[i-k].v <= y_col[i].v) and (i+k > len(y_col[1:-1]) or y_col[i].v >= y_col[i+k].v):
                    continue
                delete_indices.append(i)
                break
            if i not in delete_indices:
                for k in range(1, difference_radius):
                    if i - k > 0 and abs(y_col[i - k].v - y_col[i].v) > greatest_diff:
                        greatest_diff = abs(y_col[i - k].v - y_col[i].v)
                    if i + k < len(y_col[1:-1]) and abs(y_col[i + k].v - y_col[i].v) > greatest_diff:
                        greatest_diff = abs(y_col[i + k].v - y_col[i].v)
                if greatest_diff < minimal_absolute_difference or greatest_diff < dc.Decimal(minimal_relative_difference) * y_col[i].v or greatest_diff < dc.Decimal(minimal_difference_relative_to_biggest_absolute_extremum) * biggest_absolute_extremum:
                    delete_indices.append(i)

        res.delete(r_indices=delete_indices)



        return res

    def join(self, other_ds):
        for r_name in other_ds.get_row_names():
            self.add_row(other_ds.row(r_name))

    def delete_doubles(self, c_indices_to_check_for_doubles, epsilons, which_pick_method="mean", c_index_for_smallest_and_biggest=None):
        """takes values that have the same value in c_indices_to_check_for_doubles columns (difference < epsilons[i] for the corresponding c_indices_to_check[i]) and deletes all but one of them """
        if type(epsilons) is not list:
            epsilons = [epsilons for i in c_indices_to_check_for_doubles]
        equiv_classes = []
        for i in index_of(self.get_row_names())[:-1]:
            already_equiv = []
            for equiv_class in equiv_classes:
                already_equiv += equiv_class
            if i in already_equiv:
                continue
            for k in index_of(self.get_row_names())[i+1:]:
                r2 = self.get_row_names()[k]
                is_double_index = True
                for j in index_of(c_indices_to_check_for_doubles):
                    c = c_indices_to_check_for_doubles[j]
                    if abs(self.at(self.get_row_names()[i], c).v - self.at(r2, c).v) >= epsilons[j]:
                        is_double_index = False
                        break
                if is_double_index:
                    if len(equiv_classes) == 0 or i not in equiv_classes[-1]:
                        equiv_classes.append([i])
                    equiv_classes[-1].append(k)

        del_indices = []
        for equiv_class in equiv_classes:
            if which_pick_method == "last":
                del_indices += [ind for ind in equiv_class if ind not in del_indices][:-1]
            elif which_pick_method == "first":
                del_indices += [ind for ind in equiv_class if ind not in del_indices][1:]
            elif which_pick_method == "none":
                del_indices += [ind for ind in equiv_class if ind not in del_indices]
            elif "smallest" in which_pick_method or "biggest" in which_pick_method:
                if c_index_for_smallest_and_biggest is None:
                    c_index_for_smallest_and_biggest = which_pick_method.split(" ")[1]
                if "smallest" in which_pick_method:
                    del_indices += [ind for ind in equiv_class if ind not in del_indices]
                    equiv_class = [l for l in equiv_class if self.col(c_index_for_smallest_and_biggest)[l].v == min([self.col(c_index_for_smallest_and_biggest)[m].v for m in equiv_class])]
                else:
                    del_indices += [ind for ind in equiv_class if ind not in del_indices]
                    equiv_class = [l for l in equiv_class if self.col(c_index_for_smallest_and_biggest)[l].v == max([self.col(c_index_for_smallest_and_biggest)[m].v for m in equiv_class])]
            if which_pick_method == "weighted_mean" or "biggest" in which_pick_method or "smallest" in which_pick_method:
                del_indices += [ind for ind in equiv_class if ind not in del_indices]
                new_row = [Val(0) for i in self.get_col_names()]
                for i in index_of(self.get_col_names()):
                    new_row[i] = Val.weighted_mean([self.col(i)[l] for l in equiv_class])
                self.add_row(new_row)
        self.delete(r_indices=del_indices)

    def sort_by(self, value_function, ascending=True):
        """sorts to ascending value_function, where value_function gets row as array as only parameter"""
        tmp_name = "__tmp__"
        while tmp_name in self.get_col_names():
            tmp_name += str(rndm.randint(0, 1000000000))
        self.add_column([Val(value_function(self.row(name))) for name in self.get_row_names()], tmp_name)
        self.sort(tmp_name, ascending)
        self.delete(c_indices=[tmp_name])

class Legend_Entry:

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, new_label):
        self._label = new_label
        self.update()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color):
        self._color = new_color
        self.update()

    def update(self):
        self.patch = ptc.Patch(color=self._color, label=self.label)

    def __init__(self, name, color):
        self._label = name
        self.color = color

class Legend:
    def __init__(self, location=None, entries=None):
        if entries is None:
            entries = {}

        self.entries = entries
        self.patches = [Legend_Entry(name=entry["name"], color=entry["color"]) for entry in self.entries]
        if location is not None:
            self.location = location
        else:
            self.location = "best"

class Visualizers:
    class Dotted_line:
        def __init__(self, dataset, color="black", index_pair=None, linestyle="dashed"):
            if index_pair is None:
                index_pair = [0, 1]
            self.dataset = dataset
            self.color = color
            self.index_pair = index_pair
            self.linestyle = linestyle

    class Text:
        def __init__(self, content, position, fontsize=None, color="black", alignment="center", background_color=None):
            self.alignment = alignment
            self.content = content
            self.position = position
            self.fontsize = fontsize if fontsize is not None else plt.rcParams['font.size']
            self.color = color
            self.background_color = background_color if background_color is not None else [0, 0, 0, 0]

class Plot:

    CURVES_TOKEN = "curves"
    POINTS_TOKEN = "points"
    VISUALIZERS_TOKEN = "visualizers"
    @staticmethod
    def generate_color(index):
        golden_ratio = (1 + mt.sqrt(5)) / 2
        if index == 0:
            return [cls.hsv_to_rgb(1 / 3, 1, 1)[i] for i in range(3)]

        last_color = cls.rgb_to_hsv(Plot.generate_color(index - 1)[0], Plot.generate_color(index - 1)[1], Plot.generate_color(index - 1)[2])
        h = last_color[0] + 1 / 3
        v = last_color[2]
        s = last_color[1]
        if index % 3 == 0 and index != 0:
            h += (1 / 3) / 2 ** (mt.floor(index / 3))
            h = h - mt.floor(h)
            if index % 6 == 0:
                v -= (1 / 2) / 2 ** (mt.floor(index / 6))
            else:
                s -= (1 / 2) / 2 ** (mt.floor(1 + index / 6))

        return [cls.hsv_to_rgb(h, s, v)[i] for i in range(3)]

    @property
    def point_colors(self):
        return self._point_colors

    @point_colors.setter
    def point_colors(self, new_colors):
        self._point_colors = new_colors

    @property
    def curve_colors(self):
        return self._curve_colors

    @curve_colors.setter
    def curve_colors(self, new_colors):
        self._curve_colors = new_colors

    def _plot_curves(self):
        for i in index_of(self.curve_datasets):
            dataset = self.curve_datasets[i]
            if dataset is None:
                continue
            x_list = [val.v if isinstance(val, Val) else val for val in list(dataset.col(self.curve_column_index_pairs[i][0]))]
            y_list = [val.v if isinstance(val, Val) else val for val in list(dataset.col(self.curve_column_index_pairs[i][1]))]
            self.axes.plot(x_list, y_list,
                     color=Plot.generate_color(i) if self.curve_colors[i] is None and dataset.plot_color is None else (self.curve_colors[i] if self.curve_colors[i] is not None else dataset.plot_color))

    def _plot_visualizers(self):
        for vis in self.visualizers:
            if type(vis) is Visualizers.Dotted_line:
                self.axes.plot([val.v if isinstance(val, Val) else val for val in vis.dataset.col(vis.index_pair[0])],
                         [val.v if isinstance(val, Val) else val for val in vis.dataset.col(vis.index_pair[1])], color=vis.color,
                         linestyle=vis.linestyle)
            if type(vis) is Visualizers.Text:
                self.axes.text(vis.position[0], vis.position[1], vis.content, fontsize=vis.fontsize, color=vis.color,
                               horizontalalignment=vis.alignment, backgroundcolor=vis.background_color)

    def _plot_errors(self):
        for i in index_of(self.point_datasets):
            dataset = self.point_datasets[i]
            if dataset is None:
                continue
            err_ds = dataset.clone()

            def _fix_uncertainty(val, r, c):
                if not isinstance(val, Val):
                    return Val(str(val), "0")
                elif mt.isnan(val.e) or val.e is None:
                    val.e = 0
                return val

            err_ds.apply(_fix_uncertainty, c_indices=self.point_column_index_pairs[i])

            #to not plot lines with missing errors instead, use:
            #err_ds.filter(self.point_column_index_pairs[i][1], lambda val: not isinstance(val, Val) or mt.isnan(val.e) or val.e is None)
            #err_ds.filter(self.point_column_index_pairs[i][0], lambda val: not isinstance(val, Val) or mt.isnan(val.e) or val.e is None)

            if len(err_ds.frame.columns) > 0:
                self.axes.errorbar(x=[val.v for val in list(err_ds.col(self.point_column_index_pairs[i][0]))], y=[val.v for val in list(err_ds.col(self.point_column_index_pairs[i][1]))],
                             yerr=[self.sigma_interval[1] * val.e for val in list(err_ds.col(self.point_column_index_pairs[i][1]))],
                             xerr=[self.sigma_interval[0] * val.e for val in list(err_ds.col(self.point_column_index_pairs[i][0]))],
                             uplims=0, lolims=0, xlolims=0, xuplims=0,
                             fmt='None', ecolor=Plot.generate_color(i) if self.point_colors[i] is None and dataset.plot_color is None else (self.point_colors[i] if self.point_colors[i] is not None else dataset.plot_color))

    def _plot_points(self):
        for i in index_of(self.point_datasets):
            dataset = self.point_datasets[i]
            if dataset is None:
                continue
            if len(dataset.frame.columns) > 0:
                if dataset.plot_color is not None:
                    color = dataset.plot_color
                elif self.point_colors[i] is not None:
                    color = self.point_colors[i]
                else:
                    color = Plot.generate_color(i)
                self.axes.scatter([val.v if isinstance(val, Val) else val for val in list(dataset.col(self.point_column_index_pairs[i][0]))], [val.v if isinstance(val, Val) else val for val in list(dataset.col(self.point_column_index_pairs[i][1]))],
                            marker="x", color=color)

    def update_plt(self, update_axes=False):

        if update_axes:
            plt.clf()
            self.fig, self.axes = plt.subplots(figsize=(12, 7.5))

        self.axes.grid(which="major", linestyle="-", linewidth=1)
        self.axes.grid(which="minor", linestyle=":", linewidth=0.75)
        self.axes.xaxis.set_minor_locator(tck.AutoMinorLocator(5))
        self.axes.yaxis.set_minor_locator(tck.AutoMinorLocator(5))
        plt.figure(self.fig)
        self.axes.tick_params(axis="both", labelsize=20)

        if type(self.sigma_interval) not in [tuple, list]:
            self.sigma_interval = (self.sigma_interval, self.sigma_interval)

        for instance in self.plotting_order:
            if instance == self.CURVES_TOKEN:
                self._plot_curves()
            elif instance == self.POINTS_TOKEN:
                self._plot_points()
                self._plot_errors()
            elif instance == self.VISUALIZERS_TOKEN:
                self._plot_visualizers()

        self.axes.set_title(self.title, fontsize=40)

        if self.x_label is not None:
            self.axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            self.axes.set_ylabel(self.y_label)

        if len(self.legend.patches) > 0:
            self.axes.legend(loc=self.legend.location, handles=[entry.patch for entry in self.legend.patches], fontsize=18)

        if self.bounds["x"][0] is not None:
            self.axes.set_xlim(left=self.bounds["x"][0])
        if self.bounds["x"][1] is not None:
            self.axes.set_xlim(right=self.bounds["x"][1])
        if self.bounds["y"][0] is not None:
            self.axes.set_ylim(bottom=self.bounds["y"][0])
        if self.bounds["y"][1] is not None:
            self.axes.set_ylim(top=self.bounds["y"][1])
        self.axes.set_axisbelow(True)

    def show(self, auto_update=True):
        if auto_update:
            self.update_plt()

        self.fig.show()

    def save(self, path, dpi=None):
        self.update_plt()
        if dpi is None:
            plt.savefig(fname=path)
        else:
            plt.savefig(fname=path, dpi=dpi)

    def add_points(self, new_point_dataset, new_column_index_pair=None, color=None):
        self.point_datasets.append(new_point_dataset)
        self.point_column_index_pairs.append([0, 1] if new_column_index_pair is None else [new_column_index_pair[0] if new_column_index_pair[0] is not None else 0, new_column_index_pair[1] if new_column_index_pair[1] is not None else 1])
        self.point_colors.append(color)
        self.update_plt()

    def add_curve(self, curve_dataset, new_column_index_pair=None, color=None):
        self.curve_datasets.append(curve_dataset)
        self.curve_column_index_pairs.append((0, 1) if new_column_index_pair is None else (new_column_index_pair[0] if new_column_index_pair[0] is not None else 0, new_column_index_pair[1] if new_column_index_pair[1] is not None else 1))
        self.curve_colors.append(color)
        self.update_plt()

    def add_visualizers(self, new_visualizers):
        if type(new_visualizers) is not list:
            self.visualizers.append(new_visualizers)
        else:
            self.visualizers += new_visualizers

    def bounds_from_column(self, column_as_list, relative_margin_size=0.075, axis="x"):

        if not axis in self.bounds.keys():
            _error("keyError", "Specified axis is not a valid key of bounds dictionary. Use 'x' or 'y'. Got " + str(axis) + "instead.")

        min_val = float(min([val.v if isinstance(val, Val) else float(val) for val in column_as_list]))
        max_val = float(max([val.v if isinstance(val, Val) else float(val) for val in column_as_list]))
        self.bounds[axis][0] = min_val - relative_margin_size * (max_val - min_val)
        self.bounds[axis][1] = max_val + relative_margin_size * (max_val - min_val)

    def update_legend(self):
        entries = [{"name": self.point_datasets[i].title,
                                       "color": Plot.generate_color(i) if self.point_datasets[i].plot_color is None else self.point_datasets[i].plot_color} for i in index_of(self.point_datasets)]
        #TODO: USE self.cuvre_colors, self.point_colors!
        entries += [{"name": self.curve_datasets[i].title,
                                       "color": Plot.generate_color(i) if self.curve_datasets[i].plot_color is None else self.curve_datasets[i].plot_color} for i in index_of(self.curve_datasets)]
        self.legend = Legend(entries=entries)

    def __init__(self, point_datasets=None, curve_datasets=None, point_column_index_pairs=None, curve_column_index_pairs=None, title="Title", x_label=None, y_label=None, fig=None, axes=None):
        if curve_datasets is None:
            curve_datasets = []
        if point_datasets is None:
            point_datasets = []

        if type(point_datasets) is Dataset:
            point_datasets = [point_datasets]
        if type(curve_datasets) is Dataset:
            curve_datasets = [curve_datasets]

        if fig is None or axes is None:
            self.fig, self.axes = plt.subplots(figsize=(12, 7.5))
        else:
            self.fig = fig
            self.axes = axes

        self.title = title
        self.axes.grid(which="major", linestyle="-", linewidth=1)
        self.axes.grid(which="minor", linestyle=":", linewidth=0.75)
        self.axes.xaxis.set_minor_locator(tck.AutoMinorLocator(4))
        self.axes.yaxis.set_minor_locator(tck.AutoMinorLocator(4))
        self.point_datasets = point_datasets
        self.curve_datasets = curve_datasets
        self.sigma_interval = 1
        self.legend = Legend()
        self.bounds = {"x": [None, None],
                  "y": [None, None]}
        self.visualizers = []

        self.plotting_order = [self.CURVES_TOKEN, self.VISUALIZERS_TOKEN, self.POINTS_TOKEN]

        self.point_column_index_pairs = [[(0 if point_column_index_pairs is None or point_column_index_pairs[pcipi] is None or point_column_index_pairs[pcipi][0] is None else point_column_index_pairs[pcipi][0]), 1 if point_column_index_pairs is None or point_column_index_pairs[pcipi] is None or point_column_index_pairs[pcipi][1] is None else point_column_index_pairs[pcipi][1]] for pcipi in index_of(point_datasets)]
        self.curve_column_index_pairs = [[(0 if curve_column_index_pairs is None or curve_column_index_pairs[ccipi] is None or curve_column_index_pairs[ccipi][0] is None else curve_column_index_pairs[ccipi][0]), 1 if curve_column_index_pairs is None or curve_column_index_pairs[ccipi] is None or curve_column_index_pairs[ccipi][1] is None else curve_column_index_pairs[ccipi][1]] for ccipi in index_of(curve_datasets)]

        self.point_colors = [None for pds in self.point_datasets]
        self.curve_colors = [None for cds in self.curve_datasets]
        for i in index_of(self.point_column_index_pairs):
            for j in index_of(self.point_column_index_pairs[i]):
                if type(self.point_column_index_pairs[i][j]) is str:
                    self.point_column_index_pairs[i][j] = self.point_datasets[i].get_col_names().index(self.point_column_index_pairs[i][j])

        self.update_legend()

        # TODO:CHECK ALL LABELS
        self.x_label = x_label if x_label is not None else (self.point_datasets[0].get_col_names()[self.point_column_index_pairs[0][0]] if len(self.point_datasets) > 0 else (self.curve_datasets[0].get_col_names()[self.curve_column_index_pairs[0][0]] if len(self.curve_datasets) > 0 else None))
        self.y_label = y_label if y_label is not None else (self.point_datasets[0].get_col_names()[self.point_column_index_pairs[0][1]] if len(self.point_datasets) > 0 else (self.curve_datasets[0].get_col_names()[self.curve_column_index_pairs[0][1]] if len(self.curve_datasets) > 0 else None))
        for point_dataset in point_datasets:
            point_dataset.x_label = self.x_label
            point_dataset.y_label = self.y_label
        for curve_dataset in curve_datasets:
            curve_dataset.x_label = self.x_label
            curve_dataset.y_label = self.y_label

class Covariance_Matrix:

    def inverse(self, sigma_list):
        return np.linalg.inv(self.at(sigma_list))

    def at(self, sigma_list, as_list=False):
        arr = [[self._correlation_matrix[row_i][cell_i] * sigma_list[cell_i] * sigma_list[row_i] for cell_i in
                index_of(self._correlation_matrix[row_i])] for row_i in index_of(self._correlation_matrix)]
        if as_list:
            return arr
        return np.matrix(arr)

    def covariance_coefficient(self, variable_names, sigma):
        return self._correlation_matrix[self.variables[variable_names[0]]][self.variables[variable_names[1]]] * sigma[
            0] * sigma[1]

    def resulting_sigma(self, formula, sigma_list):
        res = 0
        for i in index_of(self.variables):
            for j in index_of(self.variables):
                res += self.covariance_coefficient((self.variables[i], self.variables[j]),
                                                   (sigma_list[i], sigma_list[j])) * smp.diff(formula.sympy,
                                                                                              self.variables[
                                                                                                  i]) * smp.diff(
                    formula.sympy, self.variables[j])

        return mt.sqrt(res)

    def __init__(self, variable_names, covariance_coefficients=None):
        self.variables = variable_names
        self._correlation_matrix = []
        if covariance_coefficients is None:
            covariance_coefficients = {}
        for var in variable_names:
            self._correlation_matrix.append([])
            for var2 in variable_names:
                try:
                    self._correlation_matrix[-1].append(covariance_coefficients[str(var) + " " + str(var2)])
                except KeyError:
                    self._correlation_matrix[-1].append(1 if str(var) is str(var2) else 0)

        self.numpy = np.matrix(self._correlation_matrix)

class Fit:
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset
        self.update()

    @staticmethod
    def fit_linear(x, y):
        one_over_sig_sqr = dc.Decimal(0)
        x_over_sig_sqr = dc.Decimal(0)
        x_sqr_over_sig_sqr = dc.Decimal(0)
        x_y_over_sig_sqr = dc.Decimal(0)
        y_over_sig_sqr = dc.Decimal(0)
        x_sum = dc.Decimal(0)
        x_square_sum = dc.Decimal(0)
        y_sum = dc.Decimal(0)
        y_square_sum = dc.Decimal(0)
        x_y_sum = dc.Decimal(0)


        for i in index_of(x):
            one_over_sig_sqr += dc.Decimal(1) / y[i].e ** dc.Decimal(2)
            x_sqr_over_sig_sqr += x[i].v ** dc.Decimal(2) / y[i].e ** dc.Decimal(2)
            x_over_sig_sqr += x[i].v / y[i].e ** dc.Decimal(2)
            x_y_over_sig_sqr += x[i].v * y[i].v / y[i].e ** dc.Decimal(2)
            y_over_sig_sqr += y[i].v / y[i].e ** dc.Decimal(2)
            x_sum += x[i].v
            y_sum += y[i].v
            x_y_sum += x[i].v * y[i].v
            x_square_sum += x[i].v**dc.Decimal(2)
            y_square_sum += y[i].v**dc.Decimal(2)

        delta = dc.Decimal(one_over_sig_sqr * x_sqr_over_sig_sqr - x_over_sig_sqr ** dc.Decimal(2))
        m = Val(0)
        b = Val(0)
        m.set(dc.Decimal(
            (dc.Decimal(1) / delta * (one_over_sig_sqr * x_y_over_sig_sqr - x_over_sig_sqr * y_over_sig_sqr))))
        m.set_err(dc.Decimal(mt.sqrt(dc.Decimal((1) / delta * one_over_sig_sqr))))
        b.set(dc.Decimal(
            dc.Decimal(1) / delta * (x_sqr_over_sig_sqr * y_over_sig_sqr - x_over_sig_sqr * x_y_over_sig_sqr)))
        b.set_err(dc.Decimal(mt.sqrt(dc.Decimal(1) / delta * x_sqr_over_sig_sqr)))

        chi_sqr = Val("0")
        for i in index_of(x):
            chi_sqr.v += dc.Decimal((dc.Decimal(1) / y[i].e * (y[i].v - m.v * x[i].v - b.v)) ** dc.Decimal(2))

        N = dc.Decimal(len(x))
        corr = (N*x_y_sum - x_sum*y_sum)/((N*x_square_sum - x_sum**dc.Decimal(2)) * (N*y_square_sum - y_sum**dc.Decimal(2)))**dc.Decimal("0.5")
        #corr is the pearson correlation coefficient as defined in Bevington: Error Analysis and Data Reduction for the physical sciences, p. 197f.
        return m, b, chi_sqr, corr

    @staticmethod
    def _fit_chi_squared_untested_algebraic(x, y, formula, estimated_parameters, precision, x_variable, fit_variables,
                                            covariance_matrix):
        D = np.matrix([[Formula(formula.variables, sympy=smp.diff(formula.at([x_variable, x[i]]), fit_variables[j])).at(
            [[fit_variables[k], estimated_parameters[k]] for k in index_of(fit_variables)], as_val=True).v for i in
                        index_of(x)] for j in index_of(fit_variables)])
        var_val_pairs = [[fit_variables[i], estimated_parameters[i]] for i in index_of(estimated_parameters)]
        f = np.matrix(
            [formula.at([cpy.deepcopy(var_val_pairs).append([x_variable, x[k]])], as_val=True).v for k in index_of(x)])

        residual = np.subtract(np.matrix([y]), f)
        delta_a = np.matmul(D.transpose(), np.linalg.inv(covariance_matrix))
        delta_a = np.matmul(delta_a, D)
        delta_a = np.matmul(np.linalg.inv(delta_a), D.transpose())
        delta_a = np.matmul(delta_a, np.linalg.inv(covariance_matrix))
        delta_a = np.matmul(delta_a, residual)

        length_delta_a = 0
        for component in delta_a[0]:
            length_delta_a += component ** 2

        length_delta_a = np.sqrt(length_delta_a)
        new_a = list(np.add(estimated_parameters, delta_a[0]))
        if length_delta_a < precision:
            M = np.matrix([y])
            M = np.subtract(M, np.matmul(D, delta_a))
            M = np.subtract(M, f)
            chi_squared = np.matmul(M.transpose(), np.linalg.inv(covariance_matrix))
            chi_squared = np.matmul(chi_squared, M)

            if abs(chi_squared - len(x) + len(fit_variables)) < precision * chi_squared:
                covariance_matrix_a = D.transpose()
                covariance_matrix_a = np.matmul(covariance_matrix_a, np.linalg.inv(covariance_matrix))
                covariance_matrix_a = np.matmul(covariance_matrix_a, D)
                covariance_matrix_a = np.linalg.inv(covariance_matrix_a)
                return new_a, covariance_matrix_a
        else:
            Fit.fit_chi_squared(x, y, formula, new_a, precision, x_variable, fit_variables, covariance_matrix)

    def fit_chi_squared(self, x, y, formula, estimated_parameters, x_variable, fit_variables, covariance_matrix, bounds, output_formula_only=False):
        if covariance_matrix is not None:
            covariance_matrix = covariance_matrix.at([y_val.e for y_val in y])
        vars = [x_variable.n]

        for variable in fit_variables:
            vars.append(variable.n)
        if estimated_parameters is None:
            estimated_parameters = [0 for p in fit_variables]
        if bounds is None:
            bounds = ([-mt.inf for p in fit_variables], [mt.inf for p in fit_variables])

        diffs = [smp.utilities.lambdify(vars, smp.diff(MatEx._substitute_constants(formula.sympy), var)) for var in vars[1:]]
        def jacobi(*args):
            res_list = []
            for x in args[0]:
                args_cpy = list(args)
                args_cpy[0] = x
                args_cpy = tuple(args_cpy)
                res = []
                for i in index_of(args[1:]):
                    res.append(diffs[i](*args_cpy))
                res_list.append(res)
            return res_list

        resulting_params, cov_mat, info_dict, mesg, ier = sp.optimize.curve_fit(
            smp.utilities.lambdify(vars, MatEx._substitute_constants(formula.sympy)), [x_val.v for x_val in x], [y_val.v for y_val in y],
            estimated_parameters, sigma=covariance_matrix, absolute_sigma=True, full_output=True, bounds=bounds, jac=jacobi, maxfev=1600)

        formula = cpy.deepcopy(formula)
        formula.set_variables([[fit_variables[i], Val(resulting_params[i], np.sqrt(np.diag(cov_mat))[i])] for i in
                               index_of(resulting_params)])
        if output_formula_only:
            return formula

        chi_squared = 0
        for i in index_of(y):
            chi_squared += (y[i].v - formula.at([[x_variable, x[i]]]).v)**2/y[i].e**2
        for i in index_of(resulting_params):
            resulting_params[i] = Val(resulting_params[i], np.sqrt(np.diag(cov_mat))[i])
            return resulting_params, chi_squared, cov_mat

    def k_fold_cross_validation(self, k, preset_folds=None):
        if __debug_extended__:
            print("\n\nPERFORMING K-FOLD TO FORMULA:")
            print(self.formula.latex)
        #preset folds is a dict {'x_lists': [_LIST_OF_LISTS_OF_X_VALS_], 'y_lists': [_LIST_OF_LISTS_OF_Y_VALS_]} where each list in 'x', 'y' represents the x or y values of a single fold
        if preset_folds is not None:
            x_lists = preset_folds["x_lists"]
            y_lists = preset_folds["y_lists"]
        else:
            split_ds = self.dataset.clone()
            x_lists = []
            y_lists = []
            n = len(split_ds.col(self.x_index))
            for j in range(k-1):
                x = []
                y = []
                for i in range(mt.floor(n/k)):
                    chosen_index = rndm.randint(0, len(split_ds.col(self.x_index))-1)
                    x.append(split_ds.col(self.x_index)[chosen_index])
                    y.append(split_ds.col(self.y_index)[chosen_index])
                    split_ds.delete(r_indices=[chosen_index])

                x_lists.append(x)
                y_lists.append(y)
            #put rest of split_ds into last fold:
            x_lists.append(split_ds.col(self.x_index))
            y_lists.append(split_ds.col(self.y_index))

        mses = []#mean squared errors
        for i in range(k):
            x = []
            y = []
            for j in [n for n in range(k) if n != i]:
                x += x_lists[j]
                y += y_lists[j]
            if __debug_extended__:
                print("STARTING", i, "-th FIT")
            formula = self.fit_chi_squared(x, y, self.fit_formula, self.estimated_parameters, self.x_variable, self.fit_variables, None, self.bounds,True)
            mse = 0
            for j in index_of(y_lists[i]):
                mse += (y_lists[i][j].v - formula.at([[self.x_variable, x_lists[i][j]]]).v) ** 2

            mse /= len(y_lists[i])
            if __debug_extended__:
                print("GOT MSE", mse)
            mses.append(mse)

        CV = 0
        for i in range(k):
            CV += mses[i]

        CV /= k

        return {"CV": CV, "MSEs": mses, "folds": {'x_lists': x_lists, 'y_lists': y_lists}}

    @staticmethod
    def chi_squared(ds, formula, x_variable, x_index=0, y_index=1):
        y = ds.col(y_index)
        x = ds.col(x_index)
        ret = 0
        for i in index_of(ds.col(x_index)):
            ret += (y[i].v - formula.at([[x_variable, x[i]]]).v) ** 2 / y[i].e ** 2
        return ret

    @staticmethod
    def reduced_chi_squared(ds, formula, x_variable, number_of_fit_parameters, x_index=0, y_index=1):
        return Fit.chi_squared(ds, formula, x_variable, x_index, y_index)/(len(ds.col(x_index)) - number_of_fit_parameters)

    def update(self):
        if self.is_linear:
            self.result["m"], self.result["b"], self.result["chi_squared"], self.result["correlation_coefficient"] = Fit.fit_linear(
                self.dataset.col(self.x_index), self.dataset.col(self.y_index))
            self.result["reduced_chi_squared"] = self.result["chi_squared"]/(len(self.dataset.col(self.x_index)) - 2)
        else:
            params, chi_squared, cov_mat = self.fit_chi_squared(self.dataset.col(self.x_index),
                                                                self.dataset.col(self.y_index), self.fit_formula,
                                                                self.estimated_parameters, self.x_variable,
                                                                self.fit_variables, self.covariance_matrix, self.bounds)
            for i in index_of(params):
                self.result[self.fit_variables[i].str] = Val(params[i], np.sqrt(np.diag(cov_mat))[i])

            self.result["chi_squared"] = chi_squared
            self.result["reduced_chi_squared"] = chi_squared / (
                    len(self.dataset.col(self.x_index)) - len(self.fit_variables))
            self.result["covariance_matrix"] = cov_mat

    def formula(self):
        if self.is_linear:
            if self.x_variable is None:
                x_name = "x" if type(self._dataset.get_col_names()[self.x_index]) is int else self._dataset.get_col_names()[
                    self.x_index]
            x_var = self.x_variable if self.x_variable is not None else Var(x_name)
            return Formula([x_var], sympy=self.result["m"].v * x_var.n +
                                          self.result["b"].v)
        else:
            var_val_pairs = [[self.fit_variables[i], self.result[self.fit_variables[i].str]]
                             for i in index_of(self.fit_variables)]
            res = self.fit_formula.clone()
            res.set_variables(var_val_pairs)
            return res

    def __init__(self, dataset, x_index=0, y_index=1, is_linear=True, fit_formula=None, x_variable=None,
                 fit_variables=None, covariance_matrix=None, estimated_parameters=None, bounds=None):

        self.result = {}
        self.fit_formula = fit_formula
        self._dataset = dataset
        self.x_index = x_index if type(x_index) is int else dataset.get_col_names().index(x_index)
        self.y_index = y_index if type(y_index) is int else dataset.get_col_names().index(y_index)
        self.is_linear = is_linear
        self.x_variable = x_variable
        self.fit_variables = fit_variables
        self.covariance_matrix = covariance_matrix
        self.estimated_parameters = estimated_parameters
        self.bounds = bounds
        self.update()