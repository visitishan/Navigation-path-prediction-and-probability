# Web Navigation Path Pattern Prediction
A library to find the Web Navigation Path Pattern Prediction.

# How to use -
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
