# Navigation Path Probability Estimation and Pattern Prediction
[![Build Status](https://img.shields.io/badge/build-passed-brightgreen.svg)]() [![Build Status](https://img.shields.io/cran/l/devtools.svg)]() [![Build Status](https://img.shields.io/badge/release-v1.0.0-green.svg)]() [![Build Status](https://img.shields.io/conda/pn/conda-forge/python.svg?maxAge=2592000)]() [![Build Status](https://img.shields.io/pypi/pyversions/Django.svg?maxAge=2592000)]()

A library to find the Probability Estimation of Navigation Paths and their Pattern Prediction.

The library helps in identifying the high probability trail path in a data. This navigation probability provides the means to analyze and predict the next link choice of unseen navigation sessions. Currently, the library allows three types of probability estimation from the path data -
- State Probability
- Transition Probability
- Path or Trail Probability

#### State Probability -
The initial probability of a state is estimated as the proportion of times the corresponding state was requested by the user. This probability is obtained by dividing the number of times a state was browsed by the total number of states browsed.

#### Transition Probability -
The probability of a transition between two states is estimated by the ratio of the number of times the sequence was visited to
the number of total paths where the from page was visited.

#### Path or Trail Probability -
The probability of a trail is estimated by the product of the initial probability of the first state in the trail and the transition probabilities of the next transitions taken in a path. The chain rule is applied in order to compute all path probabilities.


## How to use -

For the probability estimations
```
import pandas as pd
import markov_model as mm

data = {
	"other_data": [1,4,5],
    "path": [
    	["A", "B", "C", "A", "C"],
    	["B", "D", "B", "A"],
    	["A", "C", "B", "A", "D"]
    ],
    "conversions": [0, 0, 1],
}

df = pd.DataFrame(data)
print(df)
```
```
   other_data             path  conversions
0           1  [A, B, C, A, C]            0
1           4     [B, D, B, A]            0
2           5  [A, C, B, A, D]            1
```




```
# To find the state probability
state_probability = mm.state_probability(df, 'path')
print(state_probability)
```
```
  State  State_probability
0     D           0.142857
1     A           0.357143
2     C           0.214286
3     B           0.285714
```


```
# To add the start and conversion values to the path (optional)
df = mm.add_start_end(df,'path','conversions')
print(df)
```
```
   other_data                                path  conversions
0           1        [start, A, B, C, A, C, exit]            0
1           4           [start, B, D, B, A, exit]            0
2           5  [start, A, C, B, A, D, conversion]            1
```


```
# To find the transition probability
transition_df = mm.transition_probability(df, 'path')
print(transition_df)
```
```
from_sitesection to_sitesection  transition_probability
             B           exit                0.000000
             B     conversion                0.000000
             B              D                0.333333
             B              A                0.666667
             B              C                0.333333
             D           exit                0.000000
             D     conversion                0.500000
             D              B                0.500000
             D              A                0.000000
             D              C                0.000000
         start           exit                0.000000
         start     conversion                0.000000
         start              B                0.333333
         start              D                0.000000
         start              A                0.666667
         start              C                0.000000
             A           exit                0.333333
             A     conversion                0.000000
             A              B                0.333333
             A              D                0.333333
             A              C                0.666667
             C           exit                0.500000
             C     conversion                0.000000
             C              B                0.500000
             C              D                0.000000
             C              A                0.500000
```



```
# To find the path probability
path_df = mm.path_probability(df, 'path', transition_df)
print(path_df)
```
```
   other_data                                path  conversions  path_probability
0           1        [start, A, B, C, A, C, exit]            0          0.012346
1           4           [start, B, D, B, A, exit]            0          0.012346
2           5  [start, A, C, B, A, D, conversion]            1          0.024691

```


## Additional functions -

```
# To convert the path column to string data type
df = convert_to_str(df, "path")


# To convert the path column to list data type
df = convert_to_list(df, "path")
```
