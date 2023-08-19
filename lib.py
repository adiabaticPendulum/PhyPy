#ONLY CHANGE STUFF IN THIS SECTION, NOT IN THE OTHER SECTIONS. SET GLOBAL CONTENTS AND SETTINGS HERE
__debug_lib__ = __debug__   #Weather to show warnings and debug info. Default: __debug__. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application
__debug_extended__ = True   #Weather to show internal debug info (mainly for debugging the library itself).
DEC_DGTS = 128              #How many decimal digits (without rounding errors) shall be used.

#################################################################################################
#Init
ESC_STYLES = {"Error": '\033[41m\033[30m', "Error_txt": '\033[0m\033[31m', "Warning": '\033[43m\033[30m', "Warning_txt": '\033[0m\033[33m', "Default": '\033[0m', "Hacker_cliche": '\033[42m\033[30m'}

import asyncio
import pyppeteer as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decimal as dc

libs = [asyncio, pt, plt, np, pd, dc]

if DEC_DGTS <= dc.MAX_PREC:
    dc.getcontext().prec = DEC_DGTS
else:
    print(ESC_STYLES["Warning"] + "Config Warning: Passed value of" + str(DEC_DGTS) + "for DEC_DGTS exceeds allowed precission of " + str(dc.MAX_PREC) + " (might vary depending on the system). Using " + str(dc.MAX_PREC) + "instead." + ESC_STYLES["Warning"])


if __debug_lib__:
    print("Running in debug mode. Use 'python <PATH TO YOUR FILE> -oo to run in optimized mode and hide debug information")
    for lib in libs:
        try:
            print("Using", lib.__name__, lib.__version__)
        except AttributeError:
            pass
    print("\n\n####################################################################\n\n")

########################################################################################################
#internal Functions

def _warn(name, description):
    print("\n" + ESC_STYLES["Warning"] + "WARNING: "+ name + ESC_STYLES["Default"] + ESC_STYLES["Warning_txt"] + description + ESC_STYLES["Default"] + "\n")

def _error(name, description):
    print("\n" + ESC_STYLES["Error"] + "Error: " + name + ESC_STYLES["Default"] + ESC_STYLES["Error_txt"] + description +
          ESC_STYLES["Default"] + "\n")
########################################################################################################
#Fun
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

def motivate_me():#Use GPT-2 to generate and show a motivational text to convince you to proceed
    asyncio.run(motivate())


######################################################################
#Misc
def indexOf(arr):
    return range(len(arr))

#####################################################################
#Datasets and Data-Handling
class Dataset:#Object representing a full Dataset
    frame = pd.DataFrame()
    def from_dictionary(self, dictionary, items = None):    #Initialize the Dataset with a Python dictionary. Parameters: dictionary: The dictionary to read the data from. items (otional): Which items of the dictionary to use. Default is 'None', which will use all items.
        data = {}
        if items == None:
            items = list(dictionary.keys())
        if __debug_extended__:
            print("Using items:", items)

        if not isinstance(dictionary, dict):
            _error("Type Error", "Parameter 'dictionary' must be of Type <class 'dict'>, but has type " + str(type(dictionary)))
            return -1
        self.frame = pd.DataFrame(dictionary)

    def fromLists(self, lists, labels=None):
        if len(lists) != len(labels):
            if __debug_lib__:
                _warn("Parameter Length Missmatch", "Parameters 'lists' and 'labels' must have the same length but "
                                                    "have shapes (" + str(len(lists)) + ") and (" + str(len(labels)) +
                      "). Missing labels will be initialized with standard indices, missing lists with all 'None'.")
            while len(labels) < len(lists):
                labels.append(len(labels)-1)
            while len(lists) < len(labels):
                lists.append([None for i in lists[0]])
                data = {}
                for i in indexOf(labels):
                    data[labels[i]] = lists[i]
                self.frame = pd.DataFrame(data)
    #def disp(self):


##################################################
#Testing
testdict = {"a": 1, "b": 2, "d":3}

ds = Dataset()
ds.from_dictionary(testdict)
print(ds.frame)

ds.fromLists([[1, 2]], ["a", "b", "c"])
print(ds.frame)

###################################################################################################
#Best motivateMe() texts:
#I understand that your physics laboratory courses may be draining and downright boring, but don't let these setbacks deter you from your goals. Your previous progress on the lab report was phenomenal, and that is a true testament to your intelligence and hardworking nature. Though the courses may not be implemented as efficiently as they should be, do not let this dull your passions. Remain focused and committed to your goals, and you will successfully complete the lab report in no time. Keep striving for greatness!