# PhysicsLab 0.13 (Beta)
A library based on Pandas, Numpy and Matplotlib to facilitated Physics Labs at the University and evaluation of experimental results in general.

**NOTE**: Be aware that most of this is not that advanced. In fact, most of it are just some handy presets, shortcuts and small modifications, mostly designed to suit my personal needs and preferences.
There might be only small advantages over using matplotlib and pandas alone, so you might want to consider that instead. I might this library, to facilitate some work for specific use-cases (University physics labs).
While the result might be a little bit easier to use (if you understand this documentation, which probably isn't even up-to-date), however, it is certainly less versatile, less performant, less reliable, less safe and 
may lack some features. If you want to use this nevertheless, welcome aboard. 

This is a personal projects (I'm an undergrad physics student, I do this 
first and foremost to facilitate my own work. I'm no professional software-dev.), no further support, bug fixing or continuation is guaranteed whatsoever.

# Installation
To get this library to work, you need to install the following python modules:
- pyppeteer
- matplotlib
- numpy
- pandas
- latex2sympy2
- sympy 
- scipy

Additionally, you need to import (e.g. using pip):
- Jinja2

On your pc, to be able to use the latex-IO, please make sure to have MikTex installed.

# Documentation

## Constants
`__debug_lib__`

Weather to show warnings and debug info. Default: `__debug__`. Change this to False, if you want to hide the librarys internal debug information, even when you debug your application

`__debug_extended__`

Weather to show internal debug info (mainly for debugging the library itself)

`DEC_DGTS`

How many decimal digits (without rounding errors) shall be used. Default: 128

## Var
Representation of a single variable or value

### _function_ `sig_round`()
Round the value of the variable to a given number of significant values (using scientific notation).

**Parameters:**

_int_ `sig_digits`:

Number of significant digits to round to. Default: `1`

**Returns:**

List of str. First entry is the full, LaTex-formated str, second one the rounded value without error, third one the rounded absolute error, fourth one the rounded relative error only.



## Dataset
Representation of a dataset

### _Pandas.DataFrame_ `Dataset.frame`
DataFrame containing all data of the Dataset

### _function_ `Dataset.from_lists()`
Initialize the Dataset with a multidimensional list.

**Parameters:**

_List_ `lists`: List of the Lists that shall form the collumns of `Dataset.frame`

_List_ `r_names` _(optional)_: List of labels, rows shall have. Default: `None` (standard-indices)

_List_ `c_names` _(optional)_: List of the labels, the collumns/lists in `lists` shall have in `Dataset.frame`. Default: `None` (standard-indices)

_Boolean_ `strict` _(optional)_: Weather length-mismatches in the parameters shall produce errors or shall be autocorrected while producing a warning. Default: `False`

### _function_ `Dataset.from_dictionary()`
Initialize the Dataset with a Python dictionary.

**Parameters:** 

`dictionary`: The dictionary of lists to read the data from.

`items` _(otional)_: Which items (lists) of the dictionary to use. Default is `None`, which will use all items.

**Returns**:

Nothing or `-1` on failure.

### _function_ `Dataset.row(indices: _List_)`

Returns a row of the Dataset as a Pandas Series.

**Parameters:**

`indices`: Index or label of the rows to return. 

**Returns:**

Pandas Series representation of the rows with indices specified in `indices`

### _function_ <code>Dataset.disp_row(indices: _List_)</code>

Prints a row of the Dataset.

**Parameters:**

`indices`: Index or label of the rows to print. 


### _function_ <code>from_csv(path: _str_, delimiter: _str (length 1)_ (optional), c_names_from_row: _int_ (optional), c_names: _list of int_ (optional), indices_from_row _list of int_ (optional), usecols: _list of int_ (optional), userows: _list of int_ (optional), NaN_alias: _list of str_ (optional), compression: _str_, strict: _Boolean_, modify_cols: _dict_, modify_rows: _dict_)</code>
    
Initializes the Dataset from a `.csv`-file.

**Parameters:**

`path`: str that specifies the path to the `.csv`-file to be read 

`delimiter` _(optional)_: Regex that specifies the delimiter to use when reading the file. Default: `None` (uses Pandas/native Python features to auto-detect the delimiter)

`c_names_from_row` _(optional)_: int, index of row to use as column-labels. Default: `0`

`c_names` _(optional)_: List of str to use as column-labels (overriding the ones from `c_names_from_row`) Default: None (don't override existing labels)

`indices_from_row` _(optional)_: int, index of row to use as column-labels. Default: `None` (use standard-indices)

`usecols` _(optional)_: List of int, column-indices or -labels to read into Dataset. Default: `None` (use all columns)

`userows` _(optional)_: List of int, row-indices or -labels to read into Dataset. Default: `None` (use all rows)

`NaN_alias` _(optional)_: scalar, str or dict of additional entries to interpret as NaN. Default: `"NaN"`

`compression` _(optional)_: str, compression of the file to be read (e.g. "zip"). Default: `None`

`strict` _(optional)_: Weather to raise error for too long/short rows. If False, trims them and raises a warning. Default: False

`modify_cols` _(optional)_: Dict of functions to modify all values in column with corresponding key as label. The Items must have form `"<COLUMN-LABEL>": _function(entry)_`, where `_function(entry)_` has return-type `type(entry)`. Default: `{}`

`modify_rows` _(optional)_: Dict of functions to modify all values in column with corresponding key as label/index. The Items must have form `"<ROW-INDEX OR -LABEL>": _function(entry)_`, where `_function(entry)_` has return-type `type(entry)`. Default: `{}`


## Functions:

### _function_ motivateMe()
Use this, if you need motivation

### _function_ <code>indexOf(arr: _List_)</code>
**Parameters:**

_List_ `arr`:

List to return the indices of

**Returns**: List containing all list indices. Equivalent to range(len(arr))
