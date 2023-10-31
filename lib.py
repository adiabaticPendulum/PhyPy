# ONLY CHANGE STUFF IN THIS SECTION, NOT IN THE OTHER SECTIONS. SET GLOBAL CONTENTS AND SETTINGS HERE
__debug_lib__ = __debug__  # Weather to show warnings and debug info. Default: __debug__. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application
__debug_extended__ = True  # Weather to show internal debug info (mainly for debugging the library itself).

DEC_DGTS = 128  # How many decimal digits (without rounding errors) shall be used.
#################################################################################################
# Init
ESC_STYLES = {"Error": '\033[41m\033[30m', "Error_txt": '\033[0m\033[31m', "Warning": '\033[43m\033[30m',
              "Warning_txt": '\033[0m\033[33m', "Default": '\033[0m', "Hacker_cliche": '\033[42m\033[30m'}

# Note: Appart from this, you have to install Jinja2 (e.g. using pip)
import asyncio
import colorsys as cls
import math as mt
import decimal as dc
import copy as cpy
import pyppeteer as pt
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import pandas.io.formats.style as sty
import latex2sympy2 as l2s2
import sympy as smp
import re

libs = [asyncio, cls, mt, pt, plt, np, pd, dc, cpy, l2s2, tck, smp]

if DEC_DGTS <= dc.MAX_PREC:
    dc.getcontext().prec = DEC_DGTS
else:
    print(ESC_STYLES["Warning"] + "Config Warning: Passed value of" + str(
        DEC_DGTS) + "for DEC_DGTS exceeds allowed precission of " + str(
        dc.MAX_PREC) + " (might vary depending on the system). Using " + str(dc.MAX_PREC) + "instead." + ESC_STYLES[
              "Warning"])

if __debug_lib__:
    print(
        "Running in debug mode. Use 'python <PATH TO YOUR FILE> -oo to run in optimized mode and hide debug information")
    for lib in libs:
        try:
            print("Using", lib.__name__, lib.__version__)
        except AttributeError:
            pass
    print("\n\n####################################################################\n\n")

# Latex Setup:
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac, amsmath}'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['errorbar.capsize'] = 5


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


########################################################################################################
# Fun
async def motivate():
    browser = await pt.launch(headless=True)
    page = await browser.newPage()
    await page.goto("https://deepai.org/machine-learning-model/text-generator")
    await page.waitForSelector(selector=".qc-cmp2-summary-buttons")
    await page.evaluate("document.querySelector('.qc-cmp2-summary-buttons').querySelectorAll('button')[1].click()")

    prompt = "motivational text explaining that the physics laboratory courses are implemented badly, boring and exhausting. The text shall value and appreciate my previous progress on the corresponding lab report and assure me that I will manage to finish it soon."

    fn = "()=>{document.querySelector('textarea.model-input-text-input').value = '" + prompt + "';document.querySelector('button#modelSubmitButton').click();console.log('done');}"
    await page.evaluate(fn)
    await page.waitForSelector('.try-it-result-area > div > pre')
    fn = "document.querySelector('.try-it-result-area').querySelector('pre').innerText"
    res = await page.evaluate(fn)
    print(res)

    await browser.close()


def motivate_me():  # Use GPT-2 to generate and show a motivational text to convince you to proceed
    asyncio.run(motivate())


######################################################################
# Misc
def index_of(arr, start=0):
    return range(start, len(arr))


def invert_list(list):
    res = [0 for l in list]
    for i in index_of(list):
        res[len(res) - i - 1] = list[i]

    return res


#####################################################################
# Datasets and Data-Handling

class Val:  # Todo: document!

    @staticmethod
    def to_val(val, modify=lambda val: val):
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
        self._e_known_decimal_figures = 0 if "." not in str(err) else len(str(err).split('.')[1])

    def __str__(self):
        return self.sig_round(warn_for_bad_error=False)[0]
        # return str(self.v) if mt.isnan(self.e) or type(self.e) == str else self.sig_round()[0]

    def __float__(self):
        return float(self.v)

    def get(self):
        return cpy.deepcopy(self.v)

    def set(self, val):
        self.v = val

    def get_err(self):
        return cpy.deepcopy(self.e)

    def set_err(self, val):
        self.e = val

    def sig_round(self, sig_digits=1, ignore_errors=False, additional_digit=True, warn_for_bad_error=True):
        val = self.get()
        err = self.get_err()
        if err == 0 or mt.isnan(err):
            if warn_for_bad_error:
                _warn("ValueWarning",
                      "Can't sig_round() Val with error 'NaN' or '0', but got (" + str(val) + ", " + str(err) + ")")
            if val == 0:
                return ["0"]
            dec_pot = dc.Decimal(mt.floor(mt.log10(val)) - sig_digits - 1)
            val *= dc.Decimal(10 ** -dec_pot)
            val = round(val)
            return [str(val * dc.Decimal(10 ** dec_pot)) if abs(dec_pot) < 3 else str(val) + " \cdot 10\^{" + str(
                dec_pot) + "}"]

        if not ignore_errors and (mt.isnan(err) or type(val) == str or type(err) == str):
            _error("Invalid Value",
                   "Can only sig_round Vals with well-defined values and error, but got " + str(val) + ", " + str(err))
            return

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
        str_err = str(err) + ("" if dec_pot == 0 else " \cdot 10\^{" + str(dec_pot) + "}")

        val *= dc.Decimal(10 ** -dec_pot)
        val = round(val)
        str_val = str(val) + ("" if dec_pot == 0 else " \cdot 10\^{" + str(dec_pot) + "}")

        if self.v == 0:
            percent = "NaN"
        else:
            percent = dc.Decimal(100) * self.e / self.v
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
                percent = str(percent) + " \cdot 10\^{" + str(perc_pot) + "}"

        return [str_val + " \pm " + str_err + " \: (\pm " + percent + "\%)", str_val, str_err, percent]


class Var:
    n = ""
    str = ""

    def __init__(self, name):
        self.str = name
        self.n = smp.Symbol(name)

    def __str__(self):
        return self.str


class MatEx:
    variables = {}

    @property
    def sympy(self):
        return self._sympy

    @sympy.setter
    def sympy(self, new_sympy):
        self._raw_sympy = new_sympy
        self._sympy = self._raw_sympy.doit()
        self._latex = l2s2.latex(self.sympy)

    @property
    def latex(self):
        return self._latex

    @latex.setter
    def latex(self, new_latex):
        self.sympy = l2s2.latex2sympy(new_latex)

    def __init__(self, variables, latex="", sympy=None):
        for var in variables:
            if type(var) is str:
                self.variables[var] = Var(var)
            else:
                self.variables[var.n] = var
        self.sympy = sympy if sympy is not None else (None if latex == "" else l2s2.latex2sympy(latex))

    def __str__(self):
        return self.latex

    def at(self, var_val_pairs):
        tmp_sympy = self.sympy.doit()
        for var_val_pair in var_val_pairs:
            tmp_sympy = tmp_sympy.subs(var_val_pair[0], var_val_pair[1])
        return tmp_sympy


class Formula(MatEx):

    @staticmethod
    def from_mat_ex(mat_ex):
        return Formula(variables=list(mat_ex.variables.keys()), sympy=mat_ex.sympy)

    def _update_errors(self):
        _err = 0
        for var in list(self.variables.keys()):
            if "\\sigma_" not in var and "\\sigma_" + var not in list(self.variables.keys()):
                self.variables["\\sigma_" + var] = Var('\\sigma_' + var)
        for key in self.variables.keys():
            if "\\sigma_" not in key:
                _err += (smp.Derivative(self.sympy, self.variables[key].n) * self.variables["\\sigma_" + key].n) ** 2
        self.error = MatEx(variables=list(self.variables.keys()), sympy=smp.sympify(smp.sqrt(_err)))

        if __debug_extended__:
            print("Updated errors of " + self.latex + " to " + self.error.latex)

    @MatEx.sympy.setter
    def sympy(self, new_sympy):
        MatEx.sympy.fset(self, new_sympy)
        self._update_errors()

    @MatEx.latex.setter
    def latex(self, new_latex):
        MatEx.latex.fset(self, new_latex)
        self._update_errors()

    def __init__(self, variables=None, latex="", sympy=None):
        if variables is None:
            variables = []
        MatEx.__init__(self, variables, latex, sympy)

    def __str__(self):
        return MatEx.__str__(self) + " \pm " + str(self.error)

    def at(self, var_val_pairs):
        return MatEx.at(self, var_val_pairs), self.error.at(var_val_pairs)

    def to_val(self, var_val_pairs):
        at = self.at(var_val_pairs)
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
        for var_val in var_values:
            val = self.to_val([[var, var_val], ["\\sigma_" + var, 0]])
            if val.e == 0:
                val.e = dc.Decimal("NaN")
            data[val_label].append(val)

        return Dataset(dictionary=data)  # , r_names=[var, val_label]


class Dataset:  # Object representing a full Dataset
    frame = pd.DataFrame()

    def __init__(self, x_label=None, y_label=None, dictionary=None, lists=None, csv_path=None, r_names=None,
                 c_names=None, val_err_index_pairs=None):
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

    def row(self, index):
        try:
            res = self.frame.loc[index]
        except:
            res = self.frame.iloc[index]
        return res

    def col(self, index):
        try:
            res = self.frame[index]
        except:
            res = self.frame[self.get_names([0, index])[1]]
        return res

    def at(self, r_index, c_index):
        r_name = self.get_names([r_index, c_index])[0] if type(r_index) is int else r_index
        c_name = self.get_names([r_index, c_index])[1] if type(c_index) is int else c_index
        return self.frame.at[r_name, c_name]

    def set(self, r_index, c_index, value):
        r_name = self.get_names([r_index, c_index])[0] if type(r_index) is int else r_index
        c_name = self.get_names([r_index, c_index])[1] if type(c_index) is int else c_index
        self.frame.at[r_name, c_name] = value

    def disp_row(self, index):
        print(self.row([index]))

    def disp_col(self, index):
        print(self.col([index]))

    def get_names(self, location):
        return [self.frame.index[location[0]],
                self.frame.columns[location[1]]]

    def get_row_names(self):
        return self.frame.index.to_list()

    def get_col_names(self):
        return self.frame.columns.to_list()

    def apply(self, method, r_indices=None, c_indices=None):
        if r_indices is None:
            r_indices = self.get_row_names()
        if c_indices is None:
            c_indices = self.get_col_names()

        for r_index in r_indices:
            for c_index in c_indices:
                self.set(r_index, c_index, method(self.at(r_index, c_index), r_index, c_index))

    def print(self):
        print(self.frame)

    def delete(self, c_indices=None, r_indices=None):
        if c_indices is None:
            c_indices = []

        if r_indices is None:
            r_indices = []

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

        for r_name in r_names:
            self.frame.drop(index=r_name, inplace=True)

        for c_name in c_names:
            self.frame.drop(columns=c_name, inplace=True)

    def bind_error(self, value_col_index, error_col_index):
        val_col = value_col_index if type(value_col_index) is not int else self.col(value_col_index)
        err_col = error_col_index if type(error_col_index) is not int else self.col(error_col_index)

        for i in index_of(err_col):
            val_col[i].e = err_col[i].v
        self.delete(c_indices=error_col_index)

    def bind_errors(self, val_err_index_pairs):
        for pair in val_err_index_pairs:
            self.bind_error(pair[0], pair[1])

    def from_dictionary(self, dictionary, r_names=None,
                        items=None):  # Initialize the Dataset with a Python dictionary. Parameters: dictionary: The dictionary of lists to read the data from. items (otional): Which items (lists) of the dictionary to use. Default is 'None', which will use all items.
        data = {}
        if items == None:
            items = list(dictionary.keys())
        if __debug_extended__:
            print("Using items:", items)

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
                data[c_names[i]].append(Val.to_val(lists[i][j]))

        self.frame = pd.DataFrame(data, r_names)

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
                temp.loc[r][c] = Val.to_val(temp.loc[r][c])

        for k in modify_cols.keys():
            for i in index_of(temp.get[k]):
                temp.get[k][i] = modify_rows[k](temp.get[k][i])

        for k in modify_rows.keys():
            for i in index_of(temp.loc[k]):
                temp.loc[k][i] = modify_rows[k](temp.loc[k][i])

        if userows is not None:
            userows = range(shp[0])
        if usecols is not None:
            usecols = range(shp[1])

        temp = temp[[col for col in usecols]]
        self.frame = temp.loc[userows]
        self.apply(lambda obj, r_index, c_index: Val.to_val(obj))

    def filter(self, c_index, filter_method):
        col = list(self.col(c_index))
        for i in index_of(col):
            if filter_method(col[i]):
                print(i, filter_method(col[i]), col[i].e)
                self.delete(r_indices=i)
                self.filter(c_index, filter_method)
                break

    def to_csv(self, path, delimiter=";", columns=None, show_index=False):
        if columns is None:
            columns = self.get_col_names()
        self.frame.to_csv(path, sep=delimiter, index_label=self.x_label, columns=columns, index=show_index)

    def to_latex(self, show_index=False):
        styler = sty.Styler(self.frame)
        if not show_index:
            styler.hide(axis="index")

        col_frmt = "|l|"
        for col in self.get_col_names():
            col_frmt += "l|"

        ltx = styler.to_latex(position_float="centering", label="tab:my_table", caption="\\todo{caption}",
                              column_format=col_frmt)
        ltx = ltx.replace("\\begin{tabular}{" + col_frmt + "}", "\\begin{tabular}{" + col_frmt + "}" + "\n\hline")
        ltx = ltx.replace("\\\\", "\\\\ \hline", 1)
        ltx = ltx.replace("\end{tabular}", "\hline\n\end{tabular}")
        return ltx

    def clone(self):
        return cpy.deepcopy(self)


class Plot:
    point_datasets = []
    curve_datasets = []
    sigma_interval = 1

    @staticmethod
    def _color(index):
        golden_ratio = (1+mt.sqrt(5))/2
        if index == 0:
            return [cls.hsv_to_rgb(1/3, 1, 1)[i] for i in range(3)]

        last_color = cls.rgb_to_hsv(Plot._color(index - 1)[0], Plot._color(index - 1)[1], Plot._color(index - 1)[2])
        h = last_color[0] + 1/3
        v = last_color[2]
        s = last_color[1]
        if index % 3 == 0:
            h += (1/3) / 2**(mt.floor(index/3))
            h = h - mt.floor(h)
            if not index % 6 == 0:
                v -= (1/2) / 2**(mt.floor(1 + index/6))
            else:
                s -= (1/3) / 2**(mt.floor(index/6))

        return [cls.hsv_to_rgb(h, s, v)[i] for i in range(3)]

    def _update_plt(self):
        plt.figure(self.fig)

        for i in index_of(self.point_datasets):
            dataset = self.point_datasets[i]
            if dataset is None:
                continue
            err_ds = dataset.clone()
            err_ds.filter(1, lambda val: mt.isnan(val.e) or val.e is None)
            plt.errorbar(x=[val.v for val in list(err_ds.col(0))], y=[val.v for val in list(err_ds.col(1))],
                         yerr=[val.e for val in list(err_ds.col(1))], xerr=[val.e for val in list(err_ds.col(0))],
                         fmt='None', ecolor=self._color(i))
            err_ds.print()
            plt.scatter([val.v for val in list(dataset.col(0))], [val.v for val in list(dataset.col(1))],
                        marker="x", color=self._color(i))

        for i in index_of(self.curve_datasets):
            dataset = self.curve_datasets[i]
            if dataset is None:
                continue
            plt.plot([val.v for val in list(dataset.col(0))], [val.v for val in list(dataset.col(1))], color=self._color(i))

        plt.title(self.title, fontsize=30)
        if self.x_label is not None:
            plt.xlabel(self.x_label)
        if self.y_label is not None:
            plt.ylabel(self.y_label)

    def show(self):
        self._update_plt()
        plt.show()

    def save(self, path, dpi=None):
        self._update_plt()
        if dpi is None:
            plt.savefig(fname=path)
        else:
            plt.savefig(fname=path, dpi=dpi)

    def add_points(self, new_point_dataset):
        self.point_datasets.append(new_point_dataset)
        self._update_plt()

    def add_curve(self, curve_dataset):
        self.curve_datasets.append(curve_dataset)
        self._update_plt()

    def __init__(self, point_datasets=None, curve_datasets=None, title="Title", x_label=None, y_label=None):
        if curve_datasets is None:
            curve_datasets = []
        if point_datasets is None:
            point_datasets = []

        if type(point_datasets) is Dataset:
            point_datasets = [point_datasets]
        if type(curve_datasets) is Dataset:
            curve_datasets = [curve_datasets]

        self.fig, self.axes = plt.subplots(figsize=(12, 7.5))
        self.title = title
        self.axes.grid(which="major", linestyle="-", linewidth=1)
        self.axes.grid(which="minor", linestyle=":", linewidth=0.75)
        self.axes.xaxis.set_minor_locator(tck.AutoMinorLocator(4))
        self.axes.yaxis.set_minor_locator(tck.AutoMinorLocator(4))
        self.point_datasets = point_datasets
        self.curve_datasets = curve_datasets

        # TODO:CHECK ALL LABELS
        self.x_label = x_label if x_label is not None else self.point_datasets[0].x_label
        self.y_label = y_label if y_label is not None else self.point_datasets[0].y_label
        for point_dataset in point_datasets:
            point_dataset.x_label = self.x_label
            point_dataset.y_label = self.y_label
        for curve_dataset in curve_datasets:
            curve_dataset.x_label = self.x_label
            curve_dataset.y_label = self.y_label


##################################################

import random as rnd

print([cls.rgb_to_hsv(Plot._color(i)[0], Plot._color(i)[1], Plot._color(i)[2]) for i in range(10)])
# Testing
ds = [Dataset(lists=[[0, 1, 2, 3], [Val(rnd.randrange(0, 10, 1), str(rnd.randrange(0, 10, 1))) for i in range(4)]]) for j in range(9)]
plot = Plot(ds, ds, title="Test", y_label="$\\phi$")
plot.show()


###################################################################################################
# Best motivateMe() texts:
# I understand that your physics laboratory courses may be draining and downright boring, but don't let these setbacks deter you from your goals. Your previous progress on the lab report was phenomenal, and that is a true testament to your intelligence and hardworking nature. Though the courses may not be implemented as efficiently as they should be, do not let this dull your passions. Remain focused and committed to your goals, and you will successfully complete the lab report in no time. Keep striving for greatness!
