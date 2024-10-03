# PhysicsLab - Documentation

**Note**: This is not a complete documentation. Sadly, I do not have the time to create one. Instead, this is an overview over the most important features that is meant to help you get started using my library.

**Note**: Sadly, I had to remove the ability to create AI generated motivational texts since the website I used to scrape them from now seems to forbid scraping in their code of conduct. If anyone knows a free API or website I can use to legally create AI generated texts based on prompts, please contact me, so that I can revive this beloved feature.

# Introduction
## Setup
PhysicsLab is a python library based on many well renown libraries.
Apart from core python libraries, the following libraries need to be installed:

matplotlib
numpy
pandas
latex2sympy2
sympy
scipy

Apart from this, you should have a working installation of LaTex and Jinja2 installed.

All dependencies can be installed by calling:

    pip install matplotlib numpy pandas latex2sympy2 sympy scipy Jinja2
and (on debian based linux systems)

    apt install texlive-full
Note that especially the last might require a lot of memory.

## About
PhysicsLab is made by me, a physics student making his bachelors degree at the University of Göttingen. I wrote it to facilitate the everyday tasks of physics students, especially in the context of the lab courses.
This library can help you to evaluate data sets, do error propagation, create error formulas, create ready to use latex results and a lot more.

While it got quite advanced in some areas, it is noteworthy that this is not a commercial or professional product of some kind. There is no active maintainment and large parts of the code are ineffective, slow, unstable and might contain bugs. So be aware that this should not be used in a production environment. 
My hopes are simply to help some students with the tedious evaluation of lab data. Lab courses can be quite exhausting and stressful. They can be a big time factor, leading to enormous workloads and mental stress that can drive students to the borders of mental and physical disorders. I want to help students avoid the stress I needed to suffer. But I want to emphasize that every physics student should invest the time to learn to code properly. This will prove to be essential for all your further studies. So while I am happy to share this library with all of you, I highly encourage you to use it to learn to perform most of the tasks it does by yourself. Maybe do it in the semester breaks or visit dedicated courses.

At this point, credit is due to [Knetenicht](https://github.com/Knetenicht), who helped me implement some parts of this library.

## Structure and design philosophy

PhysicsLab is based on the idea to create an object orientated, well-structured library that allows for exact calculations with the most structured, clear syntax possible. For scientific calculations, it is essential to not only reduce errors, but also be able to implement complicated processes in code. Therefore, a library that allows the user to externalize a lot of that complexity to a library can help to avoid mistakes. 
In the following documentation, I will explain the most important classes and functions of the library.

----

# Configuration
In the library, there are some global variables and flags that can be used to configure the library:

### `__debug_lib__`
Weather to show warnings and debug info about the libraries classes and functions. Default: `__debug__`. Change this to `False` if you want to hide the libraries internal debug information, even when you debug your application

### `__debug_extended__`
Weather to show especially extensive internal debug info (mainly for debugging the library itself). Default: `False`

### `DEC_DGTS`
How many digits the internal `Val` class uses. Defualt: `64`

### `PRINT_OPTIONS`
A dictionary containing flags that alter the output of some functions. Currently, there is only one option: 
If `PRINT_OPTIONS[relative_uncertainties]` is set to `True`, relative uncertainties will be included in the output of some functions that print 'Val' objets, such as `Val.sig_round()`, e.g. "(2 \pm 10) \cdot 10^3 (\pm 20\percent)" instead of just ((2 \pm 10) \cdot 10^3). Default: `False`.

---- 

# Classes

## `Val`
*See also:* [Demo about `Val` objects](./demos/val_demo.py)

The most important class of this library is the `Val` class. It is used to store values with uncertainties. Most of the other functionality of this library uses the `Val` class.


---
### Operators

Most of the arithmatic operators work on `Val` objects: `Val`s can be added, subtracted, multiplied, divided and potentiated with one another. The absolute value can be taken. All of these operators will automatically apply gaussian error propagation. Note that `Val`s can not be added, multiplied, etc. with floats or other variables. If you want to perform these operators with a `Val` and an exact constant, this constant needs to implemented as a `Val` with error `0`. If you only want to affect the error *or* the value of the `Val`, use the `Val.e` and `Val.v` members. 

`Val` objects can be cast to `string`. Whenever this happens, the `Val` will be rounded to one significant digit and expressed in a LaTex compatible format (see `Val.sig_round()`)

`Val` objects can be cast to `float`. In that case, only the value is returned as a float.

---
### Constructor:
`Val(val, err)`:<br>
Creates a new Val object with given value and error.

**Paramerers**:
- `val`: The value of the Val object.
- `err`: The uncertainty/error of the Val object. Default: `"NaN"`.

*Note*: While not obligatory, it is good practice to provide the `val` and `err` arguments as strings or `decimal.Decimal` objects to not loose accuracy by casting representing them as a float first. 

--- 
### Properties:

`Val.v`: The (floating point) value of the `Val`.

---
`Val.e`: The (floating point) uncertainty of the `Val`.

---
### Member functions:

`Val.to_val(val, modify)`:<br>
Executes a callback function for a variable and returns the result as a Val object. If the callbacks return type can not be cast to Val, the original variable cast to a string will be returned instead.<br><br>
**Parameters**:
- `val`: The variable to call `modify` for.<br>
- `modify`: The callback function. Should take only one parameter, which should have the same type as val. Should return a Val object.

**Returns**:<br>
            A Val containing the return value of the callback or a string, if the callback has an incompatible return type.
----

`Val.weighted_mean(val_list)`:<br>
Calculates the weighted mean of a list of Val objects. If one of the Vals has an invalid error, the (unweighted) mean is returned as a Val object without error.

**Parameters**:
- `val_list`: A list of Val objects. Default: None

**Returns**:<br>
    The weighted mean of val_list as a Val object or the (unweighted) mean if one or more members of val_list have an invalid uncertainty, such as 'NaN' or '0'

---

`Val.sort_list(val_list, ascending=True)`:<br>
Really ineffective function to sort a list of `Val` objects in place based on their values.

**Parameters**:
- `val_list`: The list of `Val` objects to be sorted.
- `ascending`: Weather or not `val_list` should be sorted in ascending order of their values. If set to `False`, it will be sorted in descending order, instead. Default: `True`

---

`Val.sig_round(sig_digits : int, additional_digit : boolean, warn_for_bad_error: True)`<br>
Rounds the `Val` to a given number of significant digits.

**Parameters**:
- `sig_digits`: The number of significant digits to round to. Should be of type int. Default: `1`
- `additional_digit`: Weather to round to an additional digit if the first additional digit is a 1 or a 2. Default: `True`
- `warn_for_bad_error`: Weather to create a warning the Val has an invalid error, such as "NaN" or 0. Default: `True`

**Returns**:<br>
A list with the following members (in that order):
- A string describing the value with the corresponding uncertainty in the format `"([value] \pm [uncertainty]) \cdot 10^{[decimal exponent]}"`. If `PRINT_OPTIONS["relative_uncertainties"]` is `True`, the relative uncertainty will be printed as well: `"([value] \pm [uncertainty]) \cdot 10^{[decimal exponent]} \: (\pm [relative uncertainty] \\%)"`.
- A string describing the value
- A string describing the (absolute) uncertainty
- A string describing the relative uncertainty (in percent)

---

`sigma_interval(true_value)`:<br>

Returns the smallest sigma environment around a true value the `Val` is in.

**Parameters**:
- `true_value`: A float describing the true value.

**Returns**:
A float describing the offset of the true value and the `Val`s value as a fraction of the `Val`s uncertainty. E.g. `Val("1", "0.5").sigma_interval(3.25)` will yield `4.5` as $`\left|\frac{3.25 - 1}{0.5}\right| = 4.5`$.

---

`get()`:

---

`set(val)`:

---

`get_err()`:

---

`set_err(new_err)`:

---


## `Dataset`

The `Dataset` is arguably the second most important class as it represents a data set, such as the results of a measurement with varying parameters. It is used in almost all projects and provides handy functionality for reading, organizing, modifying and printing/exporting data. It is build around `pandas` functionality and stores data as `Val` objects per default.

---
### Operators

`Dataset` objects can be concatenated by adding them together using the `+` operator.

When cast to a `string`, the underlying `pandas.DataFrame` will be displayed.

---
### Constructor
`Dataset(x_label=None, y_label=None, dictionary=None, lists=None, csv_path=None, r_names=None,
                 c_names=None, val_err_index_pairs=None, title=None)`:
**Parameters**:
- `x_label`: Default: `None`
- `y_label`: Default: `None`
- `dictionary`: A dictionary containing the column names as keys and the column contents as lists as values. E.g. a dictionary `{"column_1": [1, 2, 3], "column_2": [42, 3.14, 3]}` would produce a dataset with columns `column_1` with content `1, 2, 3` and `column_2` with content `42, 3.14, 3`. If set to `None`, the Dataset contents will not be set by this property. Default: `None`
- `lists`: A multidimensional list containing the dataset contents: Each member list will form a colum, the content of which corresponds to that member lists contents. If set to `None`, the Dataset contents will not be set by this property. Default: `None`
- `csv_path`: The path to a `.csv` file that contains the dataset contents. The file contents will be read and made the dataset contents with the same column names as in the file. If set to `None`, the Dataset contents will not be set by this property. Default: `None`
- `r_names`: A list of strings describing the row names. If set to `None`, the row names will be an integer numeration of the rows. Default: `None`
- `c_names`: A list of strings describing the column names. If set to `None`, the column names are not altered.
- `val_err_index_pairs`: The dataset contents will be cast to `Val`, if possible. Per default, the uncertainty/error of the `Val`s will be set to `NaN`. By the use of this property, the error/uncertainty of members of one column can be set to the corresponding members of another column but the same row. To achieve this, this property needs to be set to a list of lists of length two. 

**Note:** All dataset contents will be cast to `Val`, if possible. This will be done by applying the `Val.to_val()` function to all contents.

---
### Properties

`Dataset.frame`
`Dataset.title`
`Dataset.plot_color`
`Dataset.x_label`
`Dataset.y_label`

---

### Member functions
`row(index)`:<br>
Returns the row at the given index. 

**Parameters**:
- `index`: Which row to return The index can be either the integer index of the row or the row name. 

**Returns**:<br>
The row contents, cast to a list.

---
`rename_rows(indices, new_names)`:

---

`rename_cols(indices, new_names)`:

---


`add_column(content, name, index=None)`:

---


`add_row(content)`:

---


`col(index)`:

---


`at(r_index, c_index)`:

---


`set(r_index, c_index, value)`:

---


`disp_row(index)`:

---


`disp_col(index)`:

---


`get_names(location)`:

---


`get_row_names()`:

---


`get_col_names()`:

---


`apply(method, r_indices=None, c_indices=None)`:

---


`print(extended=False)`:

---

`delete(c_indices=None, r_indices=None)`:<br>
Deletes the rows described by row indices in the provided list r_indices. In the same way, this deletes the columns described by the column indices found in the provided list c_indices


---


`bind_error(value_col_index, error_col_index)`:

---


`bind_errors(val_err_index_pairs)`:

---

`c_index_to_c_name(c_index)`:



---

`c_index_to_number(c_index)`:



---


`unbind_errors(c_indices)`:

---

`from_dictionary(dictionary, r_names=None, items=None)`:<br>
Initialize the Dataset with a Python dictionary.

**Parameters**:
- `dictionary`: The dictionary of lists to read the data from.
- `items`: Which items (lists) of the dictionary to use. If set to `None`, all items will be used. Default: `None`

---


`from_lists(lists, r_names=None, c_names=None, strict=False)`:

---

`from_csv(path, delimiter=None, c_names_from_row=0, c_names=None, indices_from_row=None, usecols=None, userows=None, NaN_alias="NaN", compression=None, strict=False, modify_cols={}, modify_rows={})`:

---

`filter(c_index, filter_method)`:
    """deletes every row from the dataset, for which the provided function filter_method returns True when called with the value in that row and the column described by c_index"""



---


`to_csv(path, delimiter=";", columns=None, show_index=False, exact=True)`:

---

`auto_bind_errors(error_notaition = "sigma_{")`:



---

`to_latex(show_index=False, hline_all=False)`:



---

`clone()`:



---

`move_row(old_index, new_index)`:



---

`sort(column_index, ascending=True)`:



---

`local_extrema(y_index=1, x_index=0, include_maxima=True, include_minima=True, smoothing_radius=1, difference_radius=None, minimal_absolute_difference=0, minimal_relative_difference=0, minimal_difference_relative_to_biggest_absolute_extremum=0)`:
    """smoothing radius defines, with how many neighboring values each potential extremum point is compared. minimal_absolute_difference and minimal_relative_difference are the required difference, a extremum has to have to another value in its difference_radius (absolute and relative to the extremums value)"""



---

`join(other_ds)`:



---

`delete_doubles(c_indices_to_check_for_doubles, epsilons, which_pick_method="mean", c_index_for_smallest_and_biggest=None)`:
    """takes values that have the same value in c_indices_to_check_for_doubles columns (difference < epsilons[i] for the corresponding c_indices_to_check[i]) and deletes all but one of them """



---

`sort_by(value_function, ascending=True)`:

---


## Var

For functionality like `MatEx`, which deals with unevaluated variables, `Var` objects are needed representations of variables that have no set value but act rather as a placeholder for a value. Essentially, `Var` can be thought of as an unevaluated variable. It is build around the `Sympy.symbol` class and fulfills the same purpose.


---

## MatEx

The `MatEx` class represents a mathematical expression containing one or more variables (represented by `Var` objects). It can be evaluated for any value of those `Var` objects. It is similar in function to a `Sympy` expression. `MatEx` objects are seldom used by users and are first and foremost necessary as a base class for the `Formula` class, which should be preferred in most situations. Thus, I will postpone an extensive documentation of this class to a later day (or never, depending on my time and the demand for this class' documentation).

---

## Formula

Built around the `MatEx` class, this class allows for mathematical formulas like `MatEx`, but extends the functionality to also allow for gaussian error propagation.

---

## Plot

A `Plot` object represents a line plot that visualizes a `Dataset` objects contents, including their uncertainties (if defined).

---

## Legend

---

## Legend_entry

---

## Visualizers

The `Visualizers` class contains several subclasses that represent instances that can be added to a plot. They include texts, arrows and lines.

---

## Fit

---

## Covariance_Matrix

---

## Solvers

---

# "Loose functions" and utility

### `index_of(arr, start=0)`:

### `invert_list(list)`:


### `optimal_indices(arr, target)`:

### `sort_by(list_to_sort, value_function)`:
sorts list, so that value_function evalueted by the list elements is in ascending order. So instead of sort_by(arr, foo)[i] < sort_by(arr, foo)[i+1] for all 0 <= i < len(arr) - 1, the returned list will fulfill foo(sort_by(arr, foo)[i]) < foo(sort_by(arr, foo)[i+1]) for all such i.


### `read_value(path, var_name, seperator="=", use_error=True, error_notation="sigma_", allow_pm=True, pm_notation="\pm", linebreak="\n")`:


### `dictionary_to_file(dictionary, file_path, seperator="=", linebreak="\n", error_notation="sigma_")`:
