# Representative Baseline Decision Rules

These are selected high-support leaf paths from the fitted baseline. They are descriptive model rules, not causal statements.

## Rule 1: predict `no`

- duration <= 521.500
- poutcome is not 'success'
- duration <= 205.500
- month is not 'mar'
- month is 'oct'
- duration <= 95.500
- marital is not 'divorced'

Leaf support: 76 training rows; leaf purity: 1.000; path depth: 7.

## Rule 2: predict `no`

- duration <= 521.500
- poutcome is 'success'
- duration <= 132.500
- duration > 82.500
- month is not 'sep'
- pdays > 102.500
- balance > 247.500
- month is not 'mar'
- age <= 61.000
- month is not 'oct'

Leaf support: 46 training rows; leaf purity: 1.000; path depth: 10.

## Rule 3: predict `yes`

- duration > 521.500
- duration <= 827.500
- poutcome is 'success'
- housing is 'no'
- day <= 30.500
- job is not 'entrepreneur'
- day > 1.500
- education is not 'unknown'

Leaf support: 55 training rows; leaf purity: 1.000; path depth: 8.

## Rule 4: predict `yes`

- duration > 521.500
- duration > 827.500
- contact is 'cellular'
- age <= 54.500
- poutcome is 'success'
- day > 8.000

Leaf support: 24 training rows; leaf purity: 1.000; path depth: 6.
