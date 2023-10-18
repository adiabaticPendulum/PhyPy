# ONLY CHANGE STUFF IN THIS SECTION, NOT IN THE OTHER SECTIONS. SET GLOBAL CONTENTS AND SETTINGS HERE
__debug_lib__ = __debug__  # Weather to show warnings and debug info. Default: __debug__. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application
__debug_extended__ = True  # Weather to show internal debug info (mainly for debugging the library itself).

DEC_DGTS = 128  # How many decimal digits (without rounding errors) shall be used.

#################################################################################################
# Init
ESC_STYLES = {"Error": '\033[41m\033[30m', "Error_txt": '\033[0m\033[31m', "Warning": '\033[43m\033[30m',
              "Warning_txt": '\033[0m\033[33m', "Default": '\033[0m', "Hacker_cliche": '\033[42m\033[30m'}

import asyncio
import math as mt
import decimal as dc
import copy as cpy
import pyppeteer as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import latex2sympy2 as l2s2
import sympy as smp

libs = [asyncio, mt, pt, plt, np, pd, dc, cpy, l2s2]

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


########################################################################################################
# internal Functions

def _warn(name, description):
    print("\n" + ESC_STYLES["Warning"] + "WARNING: " + name + ESC_STYLES["Default"] + " " + ESC_STYLES[
        "Warning_txt"] + description + ESC_STYLES["Default"] + "\n")


def _error(name, description):
    print(
        "\n" + ESC_STYLES["Error"] + "Error: " + name + ESC_STYLES["Default"] + " " + ESC_STYLES[
            "Error_txt"] + description +
        ESC_STYLES["Default"] + "\n")


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


def to_val(val, modify=lambda val: val):
    try:
        return Val(modify(val))
    except dc.InvalidOperation:
        return str(val)
    except:
        return Val(float(modify(val)))


#####################################################################
# Datasets and Data-Handling

class Val:  # Todo: document!

    def __init__(self, val, err="NaN"):
        self.v = dc.Decimal(val)
        self.e = dc.Decimal(err)
        self._known_decimal_figures = 0 if "." not in str(val) else len(str(val).split('.')[1])
        self._e_known_decimal_figures = 0 if "." not in str(err) else len(str(err).split('.')[1])

    def __str__(self):
        try:
            rnd = " (\\approx " + self.sig_round(ignore_errors=True)[0] + ")"
        except:
            rnd = " (No rounding available)"
        return str(self.v) + " \\pm " + str(self.e) + rnd

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

    def sig_round(self, sig_digits=1, ignore_errors=False, additional_digit=True):
        val = self.get()
        err = self.get_err()

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

    def __init__(self, name):
        self.name = smp.Symbol(name)


class MatEx:
    #TODO: TEST!
    _expr = 0

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, new_expr):
        self._expr = new_expr
        self.str = l2s2.latex(self.expr)

    def __init__(self, latex="", sympy=None):
        self.expr = sympy if sympy is not None else (l2s2.latex2sympy(latex) if not latex == "" else None)

    def at(self, var_val_pairs):
        for var_val_pair in var_val_pairs:
            self.expr = self.expr.subs(var_val_pair[0], var_val_pair[1])


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

    # def disp(self):

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
                data[item].append(to_val(val))

        if r_names is not None:
            self.frame = pd.DataFrame(data, r_names)
        else:
            self.frame = pd.DataFrame(data)

    def from_lists(self, lists, r_names=None, c_names=None, strict=False):
        if c_names is None:
            c_names = []

        if r_names is None:
            r_names = []

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
                data[c_names[i]].append(to_val(lists[i][j]))

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
                temp.loc[r][c] = to_val(temp.loc[r][c])

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
        self.apply(lambda obj, r_index, c_index: to_val(obj))


##################################################
# Testing
ds = Dataset()

ds.from_lists([[1, 2], [3, 4], [5, 6, 7]], ["a", "b", "c"], ["d", "e", "f"])
print(ds.frame)

dictionary = {
    "x": [1, 2, 3],
    "y": [4, 5, 6]
}
ds = Dataset(dictionary=dictionary)
print(ds.frame)

###################################################################################################
# Best motivateMe() texts:
# I understand that your physics laboratory courses may be draining and downright boring, but don't let these setbacks deter you from your goals. Your previous progress on the lab report was phenomenal, and that is a true testament to your intelligence and hardworking nature. Though the courses may not be implemented as efficiently as they should be, do not let this dull your passions. Remain focused and committed to your goals, and you will successfully complete the lab report in no time. Keep striving for greatness!
