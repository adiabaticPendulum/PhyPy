# PhyPy (Pre-Alpha)
A library based on Pandas, Numpy and Matplotlib to facilitated Physics Labs at the University and evaluation of experimental results in general.

This is a personal projects, no further support, bug fixing or continuation is guaranteed whatsoever.

# Documentation

## Constants
`__debug_lib__`

Weather to show warnings and debug info. Default: `__debug__`. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application

`__debug_extended__`

Weather to show internal debug info (mainly for debugging the library itself)

`DEC_DGTS`

How many decimal digits (without rounding errors) shall be used. Default: 128

## Dataset
Representation of a dataset

### _Pandas.DataFrame_ `Dataset.frame`
DataFrame containing all data of the Dataset

### _function_ `Dataset.fromLists()`
Initialize the Dataset with a multidimensional list.

**Parameters:**

_List_ `lists`: List of the Lists that shall form the collumns of `Dataset.frame`

_List_ `r_labels` _(optional)_: List of labels, rows shall have. Default: `None` (standard-indices)

_List_ `c_labels` _(optional)_: List of the labels, the collumns/lists in `lists` shall have in `Dataset.frame`. Default: `None` (standard-indices)

_Boolean_ `strict` _(optional)_: Weather length-mismatches in the parameters shall produce errors or shall be autocorrected while producing a warning. Default: `False`

### _function_ `Dataset.fromDictionary()`
Initialize the Dataset with a Python dictionary.

**Parameters:** 

`dictionary`: The dictionary of lists to read the data from.

`items` _(otional)_: Which items (lists) of the dictionary to use. Default is `None`, which will use all items.

**Returns**:

Nothing or `-1` on failure.

### _function_ `Dataset.row(indices: _List_)`

Returns a row of the Dataset as a Pandas Series.

**Parameters:**

`indices`: List of indices of the rows to return. 

**Returns:**

Pandas Series representation of the rows with indices specified in `indices`

### _function_ `Dataset.disp_row(indices: _List_)`

Prints a row of the Dataset.

**Parameters:**

`indices`: List of indices of the rows to print. 


## Functions:

### _function_ motivateMe()
Use this, if you need motivation

### _function_ indexOf(arr: _List_)
**Parameters:**

_List_ `arr`:

List to return the indices of

**Returns**: Array containing all list indices. Equivalent to range(len(arr))
