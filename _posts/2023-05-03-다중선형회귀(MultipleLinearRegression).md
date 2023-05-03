---
layout: single
title:  다중선형회귀(Multiple Linear Regression)"
categories: Python
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 2. MultipleLinear Regression


## One-Hot Encoding



```python
import matplotlib.pyplot as plt
import pandas as pd
```


```python
dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X,y 
```

<pre>
(array([[0.5, 3, 'Home'],
        [1.2, 4, 'Library'],
        [1.8, 2, 'Cafe'],
        [2.4, 0, 'Cafe'],
        [2.6, 2, 'Home'],
        [3.2, 0, 'Home'],
        [3.9, 0, 'Library'],
        [4.4, 0, 'Library'],
        [4.5, 5, 'Home'],
        [5.0, 1, 'Cafe'],
        [5.3, 2, 'Cafe'],
        [5.8, 0, 'Cafe'],
        [6.0, 3, 'Library'],
        [6.1, 1, 'Cafe'],
        [6.2, 1, 'Library'],
        [6.9, 4, 'Home'],
        [7.2, 2, 'Cafe'],
        [8.4, 1, 'Home'],
        [8.6, 1, 'Library'],
        [10.0, 0, 'Library']], dtype=object),
 array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64))
</pre>

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)
X, y
# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe
```

<pre>
(<20x22 sparse matrix of type '<class 'numpy.float64'>'
 	with 46 stored elements in Compressed Sparse Row format>,
 array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64))
</pre>
### 데이터 세트 분리



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### 학습(다중 선형 회귀)



```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
```

<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>


### 예측 값과 실제 값 비교(테스트세트)



```python
X_test, y_test
```

<pre>
(array([[0.0, 1.0, 8.6, 1],
        [0.0, 1.0, 1.2, 4],
        [0.0, 1.0, 10.0, 0],
        [1.0, 0.0, 4.5, 5]], dtype=object),
 array([ 90,   8, 100,  38], dtype=int64))
</pre>

```python
y_pred = reg.predict(X_test)
y_pred
```

<pre>
array([ 92.15457859,  10.23753043, 108.36245302,  38.14675204])
</pre>

```python
y_test
```

<pre>
array([ 90,   8, 100,  38], dtype=int64)
</pre>

```python
reg.coef_
```

<pre>
array([-5.82712824, -1.04450647, 10.40419528, -1.64200104])
</pre>

```python
reg.intercept_
```

<pre>
5.365006706544804
</pre>
### 모델평가



```python
reg.score(X_train, y_train) #훈련 세트
```

<pre>
0.9623352565265527
</pre>

```python
reg.score(X_test, y_test) # 테스트 세트
```

<pre>
0.9859956178877447
</pre>
### 다양한 평가 지표(회귀 모델)


1. MAE (Mean Absolute Error) : (실제 값과 예측 값) 차이의 절대값

1. MSE (Mean Squared Error) : 차이의 제곱

1. RSME (Root Mean Squeare Error) : 차이의 제곱에 루트

1. R2 : 결정 계수

>R2 는 1에 더 가까울수록 나머지는 0에 가까울수록 좋음



```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred) # 실제 값과 예측 값 # MAE
```

<pre>
3.2253285188287757
</pre>

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred) # 실제 값과 예측 값 # MSE
```

<pre>
19.900226981514795
</pre>

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # 실제 값과 예측 값 # RMSE
```

<pre>
4.460967045553553
</pre>

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred) # R2
## 위에 보면 rec.score 랑 똑같은데 이 평가법을 사용함
```

<pre>
0.9859956178877447
</pre>

```python
```


```python
```


```python
```


```python
```


```python
```


```python
```


```python
```
