# ONLY CHANGE STUFF IN THIS SECTION, NOT IN THE OTHER SECTIONS. SET GLOBAL CONTENTS AND SETTINGS HERE
__debug_lib__ = __debug__  # Weather to show warnings and debug info. Default: __debug__. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application
__debug_extended__ = True  # Weather to show internal debug info (mainly for debugging the library itself).

import pandas

DEC_DGTS = 128  # How many decimal digits (without rounding errors) shall be used.

#################################################################################################
# Init
ESC_STYLES = {"Error": '\033[41m\033[30m', "Error_txt": '\033[0m\033[31m', "Warning": '\033[43m\033[30m',
              "Warning_txt": '\033[0m\033[33m', "Default": '\033[0m', "Hacker_cliche": '\033[42m\033[30m'}

import asyncio
import math as mt
import pyppeteer as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decimal as dc

libs = [asyncio, mt, pt, plt, np, pd, dc]

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
def indexOf(arr, start=0):
    return range(start, len(arr))


def to_var(var, modify=lambda var: var):
    try:
        return dc.Decimal(modify(var))
    except dc.InvalidOperation:
        return str(var)


#####################################################################
# Datasets and Data-Handling

class Var:#Todo: document!
    def __init__(self, val):
        self.var = dc.Decimal(val)

    def get(self):
        return self.var


class Dataset:  # Object representing a full Dataset
    frame = pd.DataFrame()

    def from_dictionary(self, dictionary, r_labels=None,
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
        if r_labels is not None:
            self.frame = pd.DataFrame(dictionary, r_labels)
        else:
            self.frame = pd.DataFrame(dictionary)

    def from_lists(self, lists, r_labels=None, c_labels=None, strict=False):
        if len(lists) != len(c_labels):
            if strict:
                _error("Parameter Length Missmatch", "Parameters 'lists' and 'c_labels' must have the same length but "
                                                     "have shapes (" + str(len(lists)) + ") and ("
                       + str(len(c_labels)) + ").")
                return -1
            if __debug_lib__:
                _warn("Parameter Length Missmatch", "Parameters 'lists' and 'c_labels' must have the same length but "
                                                    "have shapes (" + str(len(lists)) + ") and (" + str(len(c_labels)) +
                      "). Missing c_labels will be initialized with standard indices, missing lists with all 'None'.")
            while len(c_labels) < len(lists):
                c_labels.append(len(c_labels) - 1)
            while len(lists) < len(c_labels):
                lists.append([None for i in lists[0]])
        try:
            data = {}
            for i in indexOf(c_labels):
                data[c_labels[i]] = lists[i]
            self.frame = pd.DataFrame(data)
        except ValueError:
            max_len = len(lists[0])
            for i in indexOf(lists, start=1):
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
            for i in indexOf(lists):
                while len(lists[i]) < max_len:
                    lists[i].append(dc.Decimal('NaN'))
        data = {}
        for i in indexOf(c_labels):
            data[c_labels[i]] = lists[i]
        if r_labels is not None:
            self.frame = pd.DataFrame(data, r_labels)
        else:
            self.frame = pd.DataFrame(data)

    def row(self, indices):
        return self.frame.loc[indices]

    def disp_row(self, indices):
        print(self.frame.loc[indices])

    def from_csv(self, path, delimiter=None, c_labels_from_row=0, c_labels=None, indices_from_row=None, usecols=None,
                 userows=None, NaN_alias="NaN", compression=None, strict=False, modify_cols={}, modify_rows={}):
        #TODO: TEST


        temp = pandas.read_csv(filepath_or_buffer=path, sep=delimiter, header=c_labels_from_row, names=c_labels,
                               index_col=indices_from_row, na_values=NaN_alias, na_filter=True,
                               verbose=__debug_extended__, compression=compression, quotechar="\"", comment="#",
                               on_bad_lines='error' if strict else 'warn', dtype=object)

        shp = temp.shape
        for r in range(shp[0]):
            for c in range(shp[1]):
                temp.loc[r][c] = to_var(temp.loc[r][c])

        for k in modify_cols.keys():
            for i in indexOf(temp.get[k]):
                temp.get[k][i] = modify_rows[k](temp.get[k][i])

        for k in modify_rows.keys():
            for i in indexOf(temp.loc[k]):
                temp.loc[k][i] = modify_rows[k](temp.loc[k][i])

        if userows is not None:
            userows = range(shp[0])
        if usecols is not None:
            usecols = range(shp[1])

        temp = temp[[col for col in usecols]]
        self.frame = temp.loc[userows]
    # def disp(self):


##################################################
# Testing
ds = Dataset()
print(ds.frame)

ds.from_lists([[1, 2], [3, 4], [5, 6, 7]], ["a", "b", "c"])
print("row 2:", type(ds.row(2)), "\n\n")
print("rows 1, 2:")
ds.disp_row([1, 2])

###################################################################################################
# Best motivateMe() texts:
# I understand that your physics laboratory courses may be draining and downright boring, but don't let these setbacks deter you from your goals. Your previous progress on the lab report was phenomenal, and that is a true testament to your intelligence and hardworking nature. Though the courses may not be implemented as efficiently as they should be, do not let this dull your passions. Remain focused and committed to your goals, and you will successfully complete the lab report in no time. Keep striving for greatness!
