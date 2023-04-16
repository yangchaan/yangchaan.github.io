---
layout: single
title:  "선형 회귀(Linear Regression)"
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


# 1. Linear Regression

### 공부 시간에 따른 시험 점수



```python
import matplotlib.pyplot as plt
import pandas as pd
```


```python
dataset = pd.read_csv('LinearRegressionData.csv')
```


```python
dataset.head() # 상위 5개 Data만 Display
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.8</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.6</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>



```python
dataset
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.8</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.6</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.9</td>
      <td>42</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.4</td>
      <td>48</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.3</td>
      <td>60</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5.8</td>
      <td>72</td>
    </tr>
    <tr>
      <th>12</th>
      <td>6.0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6.1</td>
      <td>68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6.2</td>
      <td>72</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.9</td>
      <td>58</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7.2</td>
      <td>76</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.4</td>
      <td>86</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8.6</td>
      <td>90</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



```python
X = dataset.iloc[:,:-1].values # 처음부터 마지막 컬럼 직전까지의 데이터( 독립 변수 = 원인 )
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터( 종속 변수 = 결과)
X,y 
```

<pre>
(array([[ 0.5],
        [ 1.2],
        [ 1.8],
        [ 2.4],
        [ 2.6],
        [ 3.2],
        [ 3.9],
        [ 4.4],
        [ 4.5],
        [ 5. ],
        [ 5.3],
        [ 5.8],
        [ 6. ],
        [ 6.1],
        [ 6.2],
        [ 6.9],
        [ 7.2],
        [ 8.4],
        [ 8.6],
        [10. ]]),
 array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64))
</pre>

```python
from sklearn.linear_model import LinearRegression
```


```python
reg = LinearRegression() # 객체 생성
reg.fit(X,y)
```

<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



```python
y_pred = reg.predict(X) # X에 대한 예측 값
y_pred
```

<pre>
array([  5.00336377,  12.31395163,  18.58016979,  24.84638795,
        26.93512734,  33.20134551,  40.51193337,  45.73378184,
        46.77815153,  52.        ,  55.13310908,  60.35495755,
        62.44369694,  63.48806663,  64.53243633,  71.84302419,
        74.97613327,  87.5085696 ,  89.59730899, 104.2184847 ])
</pre>

```python
plt.scatter(X, y, color = 'blue') # 산점도
plt.plot(X, y_pred, color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
print('9, 8, 7시간 공부했을 때 예상 점수 : ', reg.predict([[9], [8], [7]]))
```

<pre>
9, 8, 7시간 공부했을 때 예상 점수 :  [93.77478776 83.33109082 72.88739388]
</pre>

```python
reg.coef_ #기울기
```

<pre>
array([10.44369694])
</pre>

```python
reg.intercept_ # y절편
```

<pre>
-0.21848470286721522
</pre>
### 데이터 세트 분리



```python
import matplotlib.pyplot as plt
import pandas as pd
```


```python
dataset = pd.read_csv('LinearRegressionData.csv')
dataset
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.8</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.6</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.9</td>
      <td>42</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.4</td>
      <td>48</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.3</td>
      <td>60</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5.8</td>
      <td>72</td>
    </tr>
    <tr>
      <th>12</th>
      <td>6.0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6.1</td>
      <td>68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6.2</td>
      <td>72</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.9</td>
      <td>58</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7.2</td>
      <td>76</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.4</td>
      <td>86</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8.6</td>
      <td>90</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



```python
X = dataset.iloc[:, :-1].values #종속 변수
y = dataset.iloc[:, -1].values #결과 변수
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20으로 분리
```


```python
X, len(X) # 전체 데이터 X, 개수
```

<pre>
(array([[ 0.5],
        [ 1.2],
        [ 1.8],
        [ 2.4],
        [ 2.6],
        [ 3.2],
        [ 3.9],
        [ 4.4],
        [ 4.5],
        [ 5. ],
        [ 5.3],
        [ 5.8],
        [ 6. ],
        [ 6.1],
        [ 6.2],
        [ 6.9],
        [ 7.2],
        [ 8.4],
        [ 8.6],
        [10. ]]),
 20)
</pre>

```python
X_train, len(X_train) # 훈련 세트 X, 개수
```

<pre>
(array([[5.3],
        [8.4],
        [3.9],
        [6.1],
        [2.6],
        [1.8],
        [3.2],
        [6.2],
        [5. ],
        [4.4],
        [7.2],
        [5.8],
        [2.4],
        [0.5],
        [6.9],
        [6. ]]),
 16)
</pre>

```python
X_test, len(X_test)
```

<pre>
(array([[ 8.6],
        [ 1.2],
        [10. ],
        [ 4.5]]),
 4)
</pre>

```python
y, len(y)
```

<pre>
(array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64),
 20)
</pre>

```python
y_train, len(y_train)
```

<pre>
(array([60, 86, 42, 68, 22, 14, 30, 72, 58, 48, 76, 72, 26, 10, 58, 62],
       dtype=int64),
 16)
</pre>

```python
y_test, len(y_test)
```

<pre>
(array([ 90,   8, 100,  38], dtype=int64), 4)
</pre>
### 분리된 데이터를 통한 모델링



```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
```


```python
reg.fit(X_train, y_train) # 훈련 세트로 학습
```

<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>


### 데이터 시각화(훈련 세트)



```python
plt.scatter(X_train, y_train, color = 'blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours(train data)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img src="data:image/png;base64,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"/>

### 데이터 시각화(테스트 세트)



```python
plt.scatter(X_test, y_test, color = 'blue') # 산점도
plt.plot(X_test, reg.predict(X_test), color='green') # 선 그래프
plt.title('Score by hours(test data)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYUAAAEWCAYAAACJ0YulAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAnBElEQVR4nO3deXhU5dnH8e9NWAOC7BK2gLjg8moVca2lhbor9rWuqLiV1loQaxWr1uBCpS6otcUW3FCiomKLtr4qYtG6oKKCIrggyBr2PWHnfv84J4cMBBjCTM4k8/tcV67M88xZ7pnA/OZ5zpwz5u6IiIgA1Ii7ABERyRwKBRERiSgUREQkolAQEZGIQkFERCIKBRERiSgUpEozsyfN7K4UbcvNrFMqtrWnzOw9M/tBBtSRHz4vNdOw7Tpm9pWZtUj1tqXiFAqSNDM7wczeN7OVZrYsfOE6Ku66qhszOxNY7e6fhe2BZjYyRdtOW/CZWTczm5vs8u6+HngcGJCOeqRiFAqSFDNrCPwLeBhoArQGbgfWp3g/OancXiZK4l33r4CnK6OWDPAM0NvM6sRdiAQUCpKs/QHc/Vl33+zua939DXf/vHQBM/uFmU0zs9VmNtXMjgj7O5vZeDNbYWZfmtlZZdZ50sweMbNXzawY+LGZ5ZnZaDNbbGYzzazfLmprZmZjw/2+bWbtw23/1czuL7ugmb1iZv13sq0eZvatmS0P17dwvRpmdquZzTKzRWb2lJk1Cu/b7h2ymX1vZj3C2wPN7EUzG2lmq4DLzKyrmU00s1VmttDMhoTL1gZ+Arwdtk8BbgbON7M1ZjY57G9kZo+ZWZGZzTOzu0oD1cw6hc/DSjNbYmajwv53wvImh9s6f9sHb2Y5ZnZfuN4M4PRt7r+8zN94hpn9MuyvD/wfkBdue034d+xqZh+Ef/siM/tL+BgBcPe5wHLgmJ38TaQyubt+9LPLH6AhsBQYAZwKNN7m/nOBecBRgAGdgPZALWA6wQtb6QveauCAcL0ngZXA8QRvUnKBT4DbwuU7AjOAk3dQ15Ph9k4E6gAPAe+G93UF5gM1wnYzoARouYNtOcFoaG+gHbAYOCW874rwcXQEGgAvAU+H93UD5m6zre+BHuHtgcBG4OzwMdYDPgAuCe9vABwT3j4YKN5mWwOBkdv0/RP4O1AfaAF8BPwyvO9Z4JZwX3WBE7Z5jJ128nf+FfAV0JZgRPifcJ2a4f2nA/uGf+Mfhc/nETt5Ho4keMGvCeQD04D+2yzzMtAv7n/j+gl+NFKQpLj7KuAEgheI4cBiM3vZzFqGi1wF3OPuH3tgurvPInhBaAAMdvcN7v4WwQvvhWU2P8bd33P3LcChQHN3vyNcfka4vwt2Ut6/3f0dD+aobwGONbO27v4RQeB0D5e7ABjv7gt3sq3B7r7C3WcTvCAeHvb3Aoa4+wx3XwP8HrhgNw7AfuDu/3T3Le6+liAkOplZM3df4+4TwuX2Jgi5HQqf81MJXlyL3X0R8ABbn6ONBIGc5+7r3P3dJGsEOA940N3nuPsy4O6yd7r7v939u/Bv/DbwBvDDHW3M3T9x9wnuvsndvycIsh9ts9hqgsctGUChIElz92nufpm7twEOAfKAB8O72wLflbNaHjAnfMEvNYvgmESpOWVutyeYglhR+kMwymjJjkXrhy/Yy8L9QjCyuTi8fTG7nqtfUOZ2CUGglT6OWds8hpq7qKvcGkNXEkzJfWVmH5vZGWH/cmCvXWyrdARWVOY5+jvBiAHgRoJ38h+F03VXJFkjhH+vMu2yjxkzO9XMJoQfNFgBnEYwAiuXme1vZv8yswXh1Nkfy1l+L2DFbtQoaaRQkApx968Ipm4OCbvmEEwrbGs+0NbMyv5ba0cw1RRtrsztOcBMd9+7zM9e7n7aTsppW3rDzBoQTHvMD7tGAj3N7DCgM8G0S0XMJ3gxLvsYNgELgWKCaa/SGnKA5tusn3A5Ynf/1t0vJHgh/xPwYjgv/22wCWu9o3UJnqP1QLMyz1FDdz843PYCd/+Fu+cBvwSGWvKfOCqizPMZPs7Sx1UHGA3cRzAFtzfwKkEAlVcnwCME01H7uXtDgoC3bZbpDExOsj5JM4WCJMXMDjSz682sTdhuSzAFVDrt8SjwOzM70gKdwgO+HxK8aN5oZrXMrBtwJvDcDnb1EbDKzAaYWb3wwOchtvOPvp5mwcdlawN3Ah+6+xyIDmR+TDBCGB1O3VTEs8B1ZtYhDJ4/AqPcfRPwDVDXzE43s1rArQTHN3bIzC42s+bhCGpF2L3Z3TcCb5I4xbIQyC8NVncvIpi2ud/MGoYHwfc1sx+F2z639O9EMPJwYHOZbXXcSWnPA/3MrI2ZNQZuKnNf7fBxLQY2mdmpwEnb1Nm09AB8aC9gFbDGzA4Ert7meWhNEOITkIygUJBkrQaOBj604FNCE4ApwPUA7v4CMIjgI4arCd6RN3H3DcBZBHPgS4ChwKXhSGM77r6ZIDQOB2aG6zwKNCpv+dAzQAHBtNGRBPP/ZY0gOFaxJx/zfDxc/52wrnVA37DmlcCvwzrnEYTgrj6vfwrwpZmtITg4foG7rwvv+ztwSZllXwh/LzWzT8PblxK8SE8leOF/EWgV3ncUwd9pDcFB3GvdfWZ430BgRDjtdF45dQ0HXid45/4pwQF1wse5GuhHEBzLgYvC7Zfe/xVBeM4It58H/C5cbnW47VHb7O8iYER4PEgygLnrS3akejOzEwmmkfK3ObaRsczsXaCvhyewVUfhdNRk4MTwYLlkAIWCVGvhdM5zwGR3vyPuekQynaaPpNoys84E8/Wt2PopKRHZCY0UREQkopGCiIhEUn453MrUrFkzz8/Pj7sMEZEq5ZNPPlni7tueSwNU8VDIz89n4sSJcZchIlKlmNmsHd2n6SMREYkoFEREJKJQEBGRiEJBREQiCgUREYkoFEREJKJQEBGRiEJBRKQK2bxlM5f84xJGTx2dlu0rFEREqohnvniGmnfWZOTnI7l8zOVp2UeVPqNZRCQbLFizgFb3t4ra3fK7Me7ScWnZl0JBRCRDuTuXjbmMpyY/FfV985tv2K/pfmnbp0JBRCQDjf9+PD8e8eOofd9P7+P6465P+34VCiIiGaR4QzF5Q/JYtX4VAHl75TG973Tq1apXKfvXgWYRkQxx59t30uDuBlEgvH/F+8z77bxKCwTQSEFEJHZTF0/l4KEHR+2ru1zN0NOHxlKLQkFEJCabtmzi2MeOZeL8rd8Ls+SGJTTNbRpbTWmbPjKzx81skZlNKdPXxMzGmtm34e/GZe77vZlNN7OvzezkdNUlIpIJnp78NLXurBUFwujzRuMFvstAKCyE/HyoUSP4XViY2rrSOVJ4EvgL8FSZvpuAce4+2MxuCtsDzOwg4ALgYCAPeNPM9nf3zWmsT0Sk0hWtLiJvSF7U7tGxB69f/Do1bNfv0QsLoU8fKCkJ2rNmBW2AXr1SU1/aRgru/g6wbJvunsCI8PYI4Owy/c+5+3p3nwlMB7qmqzYRkcrm7vR6qVdCIEzvO52xl4xNKhAAbrllayCUKikJ+lOlso8ptHT3IgB3LzKzFmF/a2BCmeXmhn3bMbM+QB+Adu3apbFUEZHUeGvmW3R/qnvUfuDkB+h/TP/d3s7s2bvXXxGZcqDZyunz8hZ092HAMIAuXbqUu4yISCZYs2EN+9y3D8UbiwFo16gdX//ma+rWrFuh7bVrF0wZldefKpV9nsJCM2sFEP5eFPbPBdqWWa4NML+SaxMRSZmB4wey1917RYEw4coJzOo/q8KBADBoEOTmJvbl5gb9qVLZofAy0Du83RsYU6b/AjOrY2YdgP2Ajyq5NhGRPTZl0RTsduP2t28HoG/XvniBc3Sbo/d42716wbBh0L49mAW/hw1L3UFmSOP0kZk9C3QDmpnZXKAAGAw8b2ZXArOBcwHc/Uszex6YCmwCrtEnj0SkKtm0ZRNdh3flswWfRX1Lb1xKk3pNUrqfXr1SGwLbSlsouPuFO7ire3md7j4ISOEgSESkcjw56cmE7zf4x/n/4OwDz46voD2QKQeaRUSqnPmr59N6yNYPSp7S6RT+fdG/k/6IaSZSKIiI7CZ358LRFzLqy1FR34x+M+jQuEOMVaWGQkFEZDeM/W4sJ408KWr/+ZQ/0/fovjFWlFoKBRGRJKxev5rm9zZn/eb1AHRs3JGpv55KnZp1Yq4staruxJeISCW59a1baTi4YRQIH131Ed/1+67aBQJopCAiskOfL/ycw/52WNTuf3R/HjjlgRgrSj+FgojINjZu3siRw47ki0VfRH3LblxG43qNd7JW9aDpIxGRMh779DFq31U7CoSXL3gZL/CsCATQSEFEBIC5q+bS9oGtl2A7Y/8zePmClzEr73qd1ZdCQUSymrtz7gvnMnra6Khv5rUzyd87P76iYqRQEJGs9dr01zi18NSoPfS0oVx91NUxVhQ/hYKIZJ2V61bS9J6mbA6vu7l/0/354uovqJ1TO+bK4qdQEJGsUVgIV3zQlQ3NP476Jv5iIkfmHRljVZlFnz4Skaxw47DXuHi6bQ2EGT8h9x7nq/8oEMrSSEFEqrV1m9ZRb1C9xM57F0BxS0oIvvQ+nd9PUNVopCAi1VafV/okBsJrD8BAh+KWUVcqv/S+OtBIQUSqnSmLpnDoI4cm9LV7YguzZ21/zkEqv/S+OlAoiEi14e7UuCNxAuTzX33OoS0PpbAT9OkDJSVb70v1l95XB5o+EpFq4aEJDyUEwlU/uAovcA5tGYwYKuNL76sDjRREpEpbuGYh+9y/T0Lf2lvWUrdm3e2WTfeX3lcHCgURqbIO/9vhTF44OWq/etGrnLrfqTtZQ3ZFoSAiVc6/vvkXZz57ZtTukteFj3/x8U7WkGQpFESkyli7cS25f8xN6Fv4u4W0qN8ipoqqHx1oFpEq4YoxVyQEwsOnPowXuAIhxTRSEJGMNnnBZA7/++EJfVtu25J133NQWRQKIpKRtvgWcu7ISej78tdfclDzg2KqKDto+khEMs7979+fEAi/7vJrvMAVCJVAIwURyRhFq4vIG5KX0LfulnXUqVknpoqyj0JBRDLCQX89iGlLpkXt1y9+nZP2PSnGirKTQkFEYjXmqzGcPersqH1sm2N5/8r34ysoyykURCQWJRtLqP/H+gl9i29YTLPcZjFVJKADzSISg0v+cUlCIDxy+iN4gSsQMkAsIwUzuw64CnDgC+ByIBcYBeQD3wPnufvyOOoTkfT4rOgzjhh2RNTOsRw2/mGjzjnIIJUeCmbWGugHHOTua83seeAC4CBgnLsPNrObgJuAAZVdn4ikXnnnHEy7ZhoHNjswpopkR+KaPqoJ1DOzmgQjhPlAT2BEeP8I4Ox4ShORVPrTu39KCIR+XfvhBa5AyFCVPlJw93lmdh8wG1gLvOHub5hZS3cvCpcpMrNyL2hiZn2APgDt9D16Ihlr3qp5tHmgTULf+lvXUzundkwVSTIqfaRgZo0JRgUdgDygvpldnOz67j7M3bu4e5fmzZunq0wR2QOd/twpIRDevORNvMAVCFVAHAeaewAz3X0xgJm9BBwHLDSzVuEooRWwKIbaRGQPvDTtJc55/pyofWL7E3n7srdjrEh2VxyhMBs4xsxyCaaPugMTgWKgNzA4/D0mhtpEpAKKNxTT4O4GCX1Lb1xKk3pNYqpIKqrSp4/c/UPgReBTgo+j1gCGEYTBT83sW+CnYVtEMtyFoy9MCIThZw7HC1yBUEXFcp6CuxcABdt0rycYNYhIFTBx/kSOGn5U1K5Xsx7FNxfrnIMqTpe5EJHdsnnLZmremfjS8fVvvmb/pvvHVJGkki5zISJJG/TOoIRAuP7Y6/ECVyBUIxopiMguzVk5h3YPJp4XtOHWDdTKqRVTRZIuCgUR2an8B/OZtXJW1P5P7//QLb9bfAVJWikURKRcz3/5POe/eH7U7t6hO29e+maMFUllUCiISILV61fTcHDDhL5lNy6jcb3GMVUklUkHmkUkcs7z5yQEwhM9n8ALXIGQRTRSEBE+nPshxzx2TNRuVKcRywcs1zkHWUihIJLFyjvnYHrf6ezbZN+YKpK4afpIJEsNHD8wIRAGHD8AL3AFQpbTSEEky8xeOZv2D7ZP6Nv4h43UrKGXA1EoiGSV1kNaM3/1/Kj9zmXv8MP2P4yxIsk0mj4SyQLPTXkOu92iQDi106l4gSsQZDsaKYhUY6vWr6LR4EYJfSsGrKBR3UY7WEOynUYKItVUz+d6JgTCU2c/hRe4AkF2SiMFkWrmgzkfcNzjx0XtpvWasuTGJTFWJFWJQkGkmti0ZRO17ky8aumMfjPo0LhDTBVJVaTpI5Fq4A9v/SEhEG794a14gSsQZLdppCBShc1cPpOOf+6Y0KdzDmRP6F+OSBXV/N7mLCnZeqzgvSve47i2x+1kDZFd0/SRSBUz8vOR2O0WBcJZB5yFF7gCQVJCIwWRKmLlupXs/ae9E/tuWknDOg3LX0GkAjRSEKkCTis8LSEQnvnfZ/ACVyBIymmkIJLB3p39Lj98YuulKFo1aMX86+fvZA2RPaNQEMlAGzdvpPZdtRP6vr/2e9rv3X4Ha4ikhqaPRDLM79/8fUIgDPzRQLzAFQhSKTRSEMkQ05dNZ7+H90vo2/SHTeTUyImpIslGCgWRmLk7jQY3YvWG1VHfhCsncHSbo2OsSrKVpo9EYjRi0ghq3FEjCoRzOp+DF7gCQWKjkYJIDJavXU6Te5ok9K26aRV71dkrpopEAkmPFMysnpkdkM5iRLLBT5/+aUIgjPr5KLzAFQiSEZIaKZjZmcB9QG2gg5kdDtzh7mdVZKdmtjfwKHAI4MAVwNfAKCAf+B44z92XV2T7Iplo/Pfj+fGIH0ftdo3aMav/rBgrEtlestNHA4GuwHgAd59kZvl7sN+HgNfc/edmVhvIBW4Gxrn7YDO7CbgJGLAH+xDJCOWdczC7/2zaNmobU0UiO5bs9NEmd1+Zih2aWUPgROAxAHff4O4rgJ7AiHCxEcDZqdifSJxueOOGhEC468d34QWuQJCMlexIYYqZXQTkmNl+QD/g/QrusyOwGHjCzA4DPgGuBVq6exGAuxeZWYsKbl8kdt8s/YYD/pJ4CE7nHEhVkOxIoS9wMLAeeAZYCfSv4D5rAkcAj7j7D4BigqmipJhZHzObaGYTFy9eXMESRNLD3al7V92EQPjoqo/wAlcgSJWwy1AwsxzgZXe/xd2PCn9udfd1FdznXGCuu38Ytl8kCImFZtYq3GcrYFF5K7v7MHfv4u5dmjdvXsESRFLv0U8fpcYdNVi/eT0A5x98Pl7gHNX6qJgrE0neLqeP3H2zmZWYWaNUHFdw9wVmNsfMDnD3r4HuwNTwpzcwOPw9Zk/3JVIZlpYspdm9zRL61vx+DfVr14+pIpGKS/aYwjrgCzMbSzDdA4C796vgfvsCheEnj2YAlxOMWp43syuB2cC5Fdy2SKXp9mQ33p71dtQefd5o/rfz/8ZYkcieSTYU/h3+pIS7TwK6lHNX91TtQySd3pr5Ft2f2vrPdd/G+zK93/QYKxJJjaRCwd1HhO/q9w+7vnb3jekrSyQzbdi8gTp31Unom3vdXFo3bB1TRSKpldSnj8ysG/At8FdgKPCNmZ2YvrJEMk//1/onBMLd3e/GC1yBINVKstNH9wMnhQeGMbP9gWeBI9NVmEim+HrJ1xz41wMT+jbftpkaposMS/WTbCjUKg0EAHf/xsxqpakmkYzg7tS6sxabfXPU90mfTzii1RExViWSXsm+1ZloZo+ZWbfwZzjBmcgi1dLfJ/6dGnfUiAKh16G98AJXIEi1l+xI4WrgGoLLWxjwDsGxBZFqZUnJEprfm3hSZPHNxeTWyo2pIpHKlWwo1AQecvchEJ3lXGfnq4hULcc/fjzvz9l6Sa9/nv9Peh7YM8aKRCpfsqEwDugBrAnb9YA3gOPSUZRIZRr73VhOGnlS1O7crDNTr5kaY0Ui8Uk2FOq6e2kg4O5rzEzjaanS1m9aT91BdRP65v12Hnl75cVUkUj8kj3QXGxm0RE2M+sCrE1PSSLpd82/r0kIhHt/ei9e4AoEyXrJjhSuBV4ws/kEX5+ZB5yftqpE0mTq4qkcPPTghD6dcyCyVbKh0AH4AdAO+BlwDEE4iFQJ7k6NOxJf+Cf9chKH7XNYTBWJZKZk3x79wd1XAXsDPwWGAY+kqyiRVPrrR39NCITLDr8ML3AFgkg5kh0plJ7SeTrwN3cfY2YD01OSSGosLl5Mi/sSv9W15OYS6tWqF1NFIpkv2ZHCPDP7O3Ae8KqZ1dmNdUUqXdfhXRMC4ZULX8ELXIEgsgvJjhTOA04B7nP3FeHXZd6QvrJEKua16a9xauGpUft/Wv4Pk381OcaKRKqWZL9PoQR4qUy7CChKV1Eiu2vdpnXUG5Q4Cii6voh9GuwTU0UiVZOmgKTK6/NKn4RAeODkB/ACVyCIVECy00ciGWfKoikc+sihCX1bbtuCmcVUkUjVp1CQKsfd+dmonzHm6zFR3+e/+pxDWx66k7VEJBkKBalSXv32VU5/5vSo/acef+LG42+MsSKR6kWhIFXCinUraPynxlG7c7POTPrVJGrn1I6xKpHqRweaJePd8MYNCYHwaZ9PmXrNVAWCSBpopCAZ65P5n9BleJeoPeD4AQzuMTjGikSqP4WCZJwNmzdwyNBD+HbZtwDkWA5Lb1xKo7qNYq5MpPrT9JFklKEfD6XOXXWiQPi/Xv/Hpts2KRBEKolGCpIRZq2YRf5D+VH7nM7n8MK5L+icA5FKplCQWLk7Zzx7Bq9++2rUN+e6ObRp2CbGqkSyl0JBYvPK169w1nNnRe3hZw7nqiOuirEiEVEoSKVbvnY5Te5pErUPaXEIn/b5lFo5tWKsSkRAB5qlkl332nUJgTDpl5P44uovFAgiGUIjBakUH8/7mK6Pdo3aN59wM4O6D4qxIhEpT2yhYGY5wERgnrufYWZNgFFAPvA9cJ67L4+rPkmN9ZvWc9DQg5ixfAYAtXNqs/iGxTSs0zDmykSkPHFOH10LTCvTvgkY5+77AePCtlRhD3/4MHUH1Y0C4fWLX2f9resVCCIZLJaRgpm1AU4HBgG/Dbt7At3C2yOA8cCAyq5N9tzM5TPp+OeOUfu8g8/juXOe0zkHIlVAXNNHDwI3AnuV6WsZfs0n7l5kZi3KW9HM+gB9ANq1a5fmMmV3bPEtnFZ4Gq9/93rUN/e6ubRu2DrGqkRkd1T69JGZnQEscvdPKrK+uw9z9y7u3qV58+Yprk4q6p9f/ZOcO3KiQHj8rMfxAk9rIBQWQn4+1KgR/C4sTNuuRLJGHCOF44GzzOw0oC7Q0MxGAgvNrFU4SmgFLIqhNtlNy9Yuo+k9TaP24fsczkdXfZT2j5gWFkKfPlBSErRnzQraAL16pXXXItVapY8U3P337t7G3fOBC4C33P1i4GWgd7hYb2DMDjYhGaLvq30TAuHzX33OZ7/8rFLOObjllq2BUKqkJOgXkYrLpPMUBgPPm9mVwGzg3JjrkR34cO6HHPPYMVH7thNv4/Yf316pNcyevXv9IpKcWEPB3ccTfMoId18KdI+zHtm5dZvWccBfDmD2yuCVt36t+iz43QIa1G5Q6bW0axdMGZXXLyIVp8tcSFIenPAg9QbViwLhzUveZM3Na2IJBIBBgyA3N7EvNzfoF5GKy6TpI8lA3y37jk4Pd4raFx16ESN/NjL2cw5KDybfckswZdSuXRAIOsgssmcUClKuLb6Fk0eezJsz3oz65v92Pq32ahVjVYl69VIIiKSaQkG289K0lzjn+XOi9oizR3DpYZfGWJGIVBaFgkSWliyl2b3NonaXvC58cOUH1KyhfyYi2UL/2wWAa/59DUMnDo3aU66ewsEtDo6xIhGJg0Ihy30w5wOOe/y4qH17t9u57Ue3xViRiMRJoZCl1m5cS6eHOzF/9XwAGtVpxLzfzqN+7foxVyYicdJ5Clno/vfvJ/ePuVEgvHXpW6y4aYUCQUQ0Usgm3y79lv3/sn/UvvSwS3my55Oxn3MgIplDoZAFtvgWuj/VnfHfj4/6iq4vYp8G+8RXlIhkJE0fVXMvfPkCOXfkRIEw8mcj8QJXIIhIuTRSqKYWFy+mxX1bv7zu2DbH8t/L/0tOjZwYqxKRTKdQqIb6vNKH4Z8Oj9pTfz2Vzs07x1iRiFQVCoVq5L3Z73HCEydE7UE/GcTNP7w5xopEpKpRKFQDJRtL6PBQBxYVB99g2qReE2b3n62PmIrIbtOB5irunvfuof4f60eBML73eJbeuFSBICIVopFCFfXN0m844C8HRO0rDr+Cx3o+FmNFIlIdKBSqmM1bNtNtRDfenf1u1Lfg+gW0bNAyvqJEpNrQ9FEVMmrKKGreWTMKhGfPeRYvcAWCiKSMRgpVwKLiRbS8b+sL/wntTmB87/E650BEUk6hkOGuHHMlj096PGp/dc1XHNDsgJ2sISJScQqFDPXfWf/lxCdPjNqDuw9mwAkDYqxIRLKBQiHDlGwsoe0DbVm2dhkALeq3YOa1M8mtlRtzZSKSDXSgOYPc/d+7qf/H+lEg/Pfy/7LwdwsVCCJSaTRSyADTFk/joKEHRe1fHPELhp05LMaKRCRbKRRitHnLZk544gQmzJ0Q9S363SKa128eY1Uiks00fRSTZ754hpp31owCYdTPR+EFrkAQkVhppFDJFqxZQKv7W0XtbvndGHfpOGqY8llE4qdQqCTuzmVjLuOpyU9Ffd/85hv2a7pfjFWJiCRSKFSC/8z8Dz956idR+54e93DD8TfEWJGISPkqPRTMrC3wFLAPsAUY5u4PmVkTYBSQD3wPnOfuyyu7vlQq3lBM3pA8Vq1fBUDeXnlM7zuderXqxVyZiEj54pjI3gRc7+6dgWOAa8zsIOAmYJy77weMC9tV1p1v30mDuxtEgfDeFe8x77fzFAgiktEqfaTg7kVAUXh7tZlNA1oDPYFu4WIjgPFAlbuuw5eLvuSQRw6J2ld3uZqhpw+NsSIRkeTFekzBzPKBHwAfAi3DwMDdi8ysRZy17a5NWzZx7GPHMnH+xKhv8Q2LaZbbLMaqRER2T2yfgzSzBsBooL+7r9qN9fqY2UQzm7h48eL0Fbgbnp78NLXurBUFwovnvogXuAJBRKqcWEYKZlaLIBAK3f2lsHuhmbUKRwmtgEXlrevuw4BhAF26dPFKKXgHilYXkTckL2r36NiD1y9+XecciEiVVemvXmZmwGPANHcfUuaul4He4e3ewJjKri1Z7k6vl3olBML0vtMZe8lYBYKIVGlxjBSOBy4BvjCzSWHfzcBg4HkzuxKYDZwbQ227NG7GOHo83SNqDzlpCNcde12MFYmIpE4cnz56F7Ad3N29MmvZHavXr2af+/ehZGMJAG0btuWbvt9Qt2bdmCsTEUkdzXUkYeD4gTQc3DAKhA+u/IDZ181WIIhItaPLXOzElEVTOPSRQ6P2b476DQ+f9nCMFYmIpJdCoRybtmyi6/CufLbgs6hvyQ1LaJrbNMaqRETST9NH23jisyeodWetKBBeOu8lvMAVCCKSFTRSCM1fPZ/WQ1pH7ZP3PZlXe72qj5iKSFbJ+lBwdy4cfSGjvhwV9X3X7zs6Nu4YY1UiIvHI6lAY+91YThp5UtR+6JSH6Hd0vxgrEhGJV9aGQqPBjaLLWnfYuwPTrplGnZp1Yq5KRCReWRkKXy/5OgqEj676iKNaHxVzRSIimSErQ+GAZgdQfHMxubVy4y5FRCSjZO1HaxQIIiLby9pQEBGR7SkUREQkolAQEZGIQkFERCIKBRERiWRlKBQWQn4+1KgR/C4sjLsiEZHMkHXnKRQWQp8+UBJ8Xw6zZgVtgF694qtLRCQTZN1I4ZZbtgZCqZKSoF9EJNtlXSjMnr17/SIi2STrQqFdu93rFxHJJlkXCoMGQe42V7jIzQ36RUSyXdaFQq9eMGwYtG8PZsHvYcN0kFlEBLLw00cQBIBCQERke1k3UhARkR1TKIiISEShICIiEYWCiIhEFAoiIhIxd4+7hgozs8XArLjrSFIzYEncRWQYPSfl0/OyPT0n29uT56S9uzcv744qHQpViZlNdPcucdeRSfSclE/Py/b0nGwvXc+Jpo9ERCSiUBARkYhCofIMi7uADKTnpHx6Xran52R7aXlOdExBREQiGimIiEhEoSAiIhGFQpqZWVsz+4+ZTTOzL83s2rhryhRmlmNmn5nZv+KuJROY2d5m9qKZfRX+ezk27priZmbXhf9vppjZs2ZWN+6a4mBmj5vZIjObUqaviZmNNbNvw9+NU7EvhUL6bQKud/fOwDHANWZ2UMw1ZYprgWlxF5FBHgJec/cDgcPI8ufGzFoD/YAu7n4IkANcEG9VsXkSOGWbvpuAce6+HzAubO8xhUKauXuRu38a3l5N8B+9dbxVxc/M2gCnA4/GXUsmMLOGwInAYwDuvsHdV8RaVGaoCdQzs5pALjA/5npi4e7vAMu26e4JjAhvjwDOTsW+FAqVyMzygR8AH8ZcSiZ4ELgR2BJzHZmiI7AYeCKcUnvUzOrHXVSc3H0ecB8wGygCVrr7G/FWlVFaunsRBG8+gRap2KhCoZKYWQNgNNDf3VfFXU+czOwMYJG7fxJ3LRmkJnAE8Ii7/wAoJkXTAVVVOEfeE+gA5AH1zezieKuq/hQKlcDMahEEQqG7vxR3PRngeOAsM/seeA74iZmNjLek2M0F5rp76SjyRYKQyGY9gJnuvtjdNwIvAcfFXFMmWWhmrQDC34tSsVGFQpqZmRHME09z9yFx15MJ3P337t7G3fMJDhy+5e5Z/Q7Q3RcAc8zsgLCrOzA1xpIywWzgGDPLDf8fdSfLD75v42Wgd3i7NzAmFRutmYqNyE4dD1wCfGFmk8K+m9391fhKkgzVFyg0s9rADODymOuJlbt/aGYvAp8SfIrvM7L0chdm9izQDWhmZnOBAmAw8LyZXUkQoOemZF+6zIWIiJTS9JGIiEQUCiIiElEoiIhIRKEgIiIRhYKIiEQUCiJJMLP8sleoFKmuFAoiMQkv8iaSURQKIsnLMbPh4fX93zCzemZ2uJlNMLPPzewfpde0N7PxZtYlvN0svKQHZnaZmb1gZq8Ab5hZKzN7x8wmhd8Z8MP4Hp6IQkFkd+wH/NXdDwZWAOcATwED3P1/gC8IzjTdlWOB3u7+E+Ai4HV3P5zgOxQmpb5skeRp+CqSvJnuPim8/QmwL7C3u78d9o0AXkhiO2PdvfTa+B8Dj4cXTfxnme2LxEIjBZHkrS9zezOw906W3cTW/1/bfoVkcemN8MtTTgTmAU+b2aV7XqZIxSkURCpuJbC8zHGAS4DSUcP3wJHh7Z/vaANm1p7guyWGE1xNN9svly0x0/SRyJ7pDfzNzHJJvLLpfQRXsLwEeGsn63cDbjCzjcAaQCMFiZWukioiIhFNH4mISEShICIiEYWCiIhEFAoiIhJRKIiISEShICIiEYWCiIhE/h/7XxGMNiXjGQAAAABJRU5ErkJggg=="/>


```python
reg.coef_
```

<pre>
array([10.49161294])
</pre>

```python
reg.intercept_
```

<pre>
0.6115562905169796
</pre>
### 모델 평가



```python
reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가
```

<pre>
0.9727616474310156
</pre>

```python
reg.score(X_train, y_train) # 훈련 세트를 통한 모델 평가
```

<pre>
0.9356663661221668
</pre>
## 경하 하강법(Gradient Descent)



```python
from sklearn.linear_model import SGDRegressor # SGD : Stocahstic Gradient Descent 확률적 경하 하강법
sr = SGDRegressor()
sr.fit(X_train, y_train)
```

<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SGDRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">SGDRegressor</label><div class="sk-toggleable__content"><pre>SGDRegressor()</pre></div></div></div></div></div>



```python
plt.scatter(X_train, y_train, color = 'blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours(train data, SGD)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
sr.coef_, sr.intercept_
```

<pre>
(array([10.35260639]), array([1.5578431]))
</pre>

```python
sr.score(X_train, y_train) # 훈련 세트를 통한 모델 평가
```

<pre>
0.9353426764545777
</pre>

```python
sr.score(X_test, y_test) # 테스트 세트를 통한 모델 평가
```

<pre>
0.9709924586604373
</pre>
max_iter : 훈련 세트 반복 횟수 (Epoch 횟수)



eta0 : 학습률 (learning rate)



```python
# 지수 표기법
# 1e-3 : 0.001(10^-3)
# 1e+3 : 0.001(10^3)

sr = SGDRegressor(max_iter=100, eta0=1e-3, random_state=0, verbose=1) # max_iter 값과 eta0값을 바꿔서 해보면 이해됨
sr.fit(X_train, y_train)
```

<pre>
-- Epoch 1
Norm: 2.40, NNZs: 1, Bias: 0.442470, T: 16, Avg. loss: 1181.034371
Total training time: 0.00 seconds.
-- Epoch 2
Norm: 3.84, NNZs: 1, Bias: 0.697455, T: 32, Avg. loss: 754.011321
Total training time: 0.00 seconds.
-- Epoch 3
Norm: 4.89, NNZs: 1, Bias: 0.881472, T: 48, Avg. loss: 520.842928
Total training time: 0.00 seconds.
-- Epoch 4
Norm: 5.70, NNZs: 1, Bias: 1.023556, T: 64, Avg. loss: 374.527388
Total training time: 0.00 seconds.
-- Epoch 5
Norm: 6.34, NNZs: 1, Bias: 1.137258, T: 80, Avg. loss: 277.717040
Total training time: 0.00 seconds.
-- Epoch 6
Norm: 6.88, NNZs: 1, Bias: 1.230635, T: 96, Avg. loss: 210.603548
Total training time: 0.00 seconds.
-- Epoch 7
Norm: 7.32, NNZs: 1, Bias: 1.308149, T: 112, Avg. loss: 162.433366
Total training time: 0.00 seconds.
-- Epoch 8
Norm: 7.69, NNZs: 1, Bias: 1.372847, T: 128, Avg. loss: 127.468199
Total training time: 0.00 seconds.
-- Epoch 9
Norm: 8.01, NNZs: 1, Bias: 1.427757, T: 144, Avg. loss: 101.814505
Total training time: 0.00 seconds.
-- Epoch 10
Norm: 8.28, NNZs: 1, Bias: 1.474953, T: 160, Avg. loss: 82.674196
Total training time: 0.00 seconds.
-- Epoch 11
Norm: 8.51, NNZs: 1, Bias: 1.515486, T: 176, Avg. loss: 68.085082
Total training time: 0.00 seconds.
-- Epoch 12
Norm: 8.71, NNZs: 1, Bias: 1.549985, T: 192, Avg. loss: 57.005190
Total training time: 0.00 seconds.
-- Epoch 13
Norm: 8.88, NNZs: 1, Bias: 1.580062, T: 208, Avg. loss: 48.534157
Total training time: 0.00 seconds.
-- Epoch 14
Norm: 9.04, NNZs: 1, Bias: 1.606388, T: 224, Avg. loss: 41.986284
Total training time: 0.00 seconds.
-- Epoch 15
Norm: 9.17, NNZs: 1, Bias: 1.629265, T: 240, Avg. loss: 36.843002
Total training time: 0.00 seconds.
-- Epoch 16
Norm: 9.29, NNZs: 1, Bias: 1.649170, T: 256, Avg. loss: 32.831436
Total training time: 0.00 seconds.
-- Epoch 17
Norm: 9.39, NNZs: 1, Bias: 1.666531, T: 272, Avg. loss: 29.701149
Total training time: 0.00 seconds.
-- Epoch 18
Norm: 9.48, NNZs: 1, Bias: 1.682057, T: 288, Avg. loss: 27.231481
Total training time: 0.00 seconds.
-- Epoch 19
Norm: 9.56, NNZs: 1, Bias: 1.695737, T: 304, Avg. loss: 25.239918
Total training time: 0.00 seconds.
-- Epoch 20
Norm: 9.63, NNZs: 1, Bias: 1.707648, T: 320, Avg. loss: 23.666198
Total training time: 0.00 seconds.
-- Epoch 21
Norm: 9.70, NNZs: 1, Bias: 1.717986, T: 336, Avg. loss: 22.423381
Total training time: 0.00 seconds.
-- Epoch 22
Norm: 9.75, NNZs: 1, Bias: 1.727520, T: 352, Avg. loss: 21.423074
Total training time: 0.00 seconds.
-- Epoch 23
Norm: 9.81, NNZs: 1, Bias: 1.735911, T: 368, Avg. loss: 20.599154
Total training time: 0.00 seconds.
-- Epoch 24
Norm: 9.85, NNZs: 1, Bias: 1.743059, T: 384, Avg. loss: 19.946664
Total training time: 0.00 seconds.
-- Epoch 25
Norm: 9.89, NNZs: 1, Bias: 1.749411, T: 400, Avg. loss: 19.425241
Total training time: 0.00 seconds.
-- Epoch 26
Norm: 9.93, NNZs: 1, Bias: 1.755224, T: 416, Avg. loss: 19.006171
Total training time: 0.00 seconds.
-- Epoch 27
Norm: 9.96, NNZs: 1, Bias: 1.760444, T: 432, Avg. loss: 18.649850
Total training time: 0.00 seconds.
-- Epoch 28
Norm: 9.99, NNZs: 1, Bias: 1.764855, T: 448, Avg. loss: 18.364863
Total training time: 0.00 seconds.
-- Epoch 29
Norm: 10.02, NNZs: 1, Bias: 1.768742, T: 464, Avg. loss: 18.136890
Total training time: 0.00 seconds.
-- Epoch 30
Norm: 10.04, NNZs: 1, Bias: 1.772303, T: 480, Avg. loss: 17.954396
Total training time: 0.00 seconds.
-- Epoch 31
Norm: 10.06, NNZs: 1, Bias: 1.775650, T: 496, Avg. loss: 17.789278
Total training time: 0.00 seconds.
-- Epoch 32
Norm: 10.08, NNZs: 1, Bias: 1.778365, T: 512, Avg. loss: 17.661331
Total training time: 0.01 seconds.
-- Epoch 33
Norm: 10.10, NNZs: 1, Bias: 1.780615, T: 528, Avg. loss: 17.556661
Total training time: 0.01 seconds.
-- Epoch 34
Norm: 10.11, NNZs: 1, Bias: 1.782892, T: 544, Avg. loss: 17.473956
Total training time: 0.01 seconds.
-- Epoch 35
Norm: 10.13, NNZs: 1, Bias: 1.784920, T: 560, Avg. loss: 17.398214
Total training time: 0.01 seconds.
-- Epoch 36
Norm: 10.14, NNZs: 1, Bias: 1.786514, T: 576, Avg. loss: 17.338250
Total training time: 0.01 seconds.
-- Epoch 37
Norm: 10.15, NNZs: 1, Bias: 1.787809, T: 592, Avg. loss: 17.288143
Total training time: 0.01 seconds.
-- Epoch 38
Norm: 10.16, NNZs: 1, Bias: 1.789211, T: 608, Avg. loss: 17.250074
Total training time: 0.01 seconds.
-- Epoch 39
Norm: 10.17, NNZs: 1, Bias: 1.790389, T: 624, Avg. loss: 17.214517
Total training time: 0.01 seconds.
-- Epoch 40
Norm: 10.18, NNZs: 1, Bias: 1.791332, T: 640, Avg. loss: 17.184784
Total training time: 0.01 seconds.
-- Epoch 41
Norm: 10.19, NNZs: 1, Bias: 1.792025, T: 656, Avg. loss: 17.159879
Total training time: 0.01 seconds.
-- Epoch 42
Norm: 10.20, NNZs: 1, Bias: 1.792728, T: 672, Avg. loss: 17.143211
Total training time: 0.01 seconds.
-- Epoch 43
Norm: 10.21, NNZs: 1, Bias: 1.793462, T: 688, Avg. loss: 17.122124
Total training time: 0.01 seconds.
-- Epoch 44
Norm: 10.21, NNZs: 1, Bias: 1.793882, T: 704, Avg. loss: 17.109227
Total training time: 0.01 seconds.
-- Epoch 45
Norm: 10.22, NNZs: 1, Bias: 1.794138, T: 720, Avg. loss: 17.096271
Total training time: 0.01 seconds.
-- Epoch 46
Norm: 10.22, NNZs: 1, Bias: 1.794484, T: 736, Avg. loss: 17.089082
Total training time: 0.01 seconds.
-- Epoch 47
Norm: 10.23, NNZs: 1, Bias: 1.794875, T: 752, Avg. loss: 17.076854
Total training time: 0.01 seconds.
-- Epoch 48
Norm: 10.23, NNZs: 1, Bias: 1.794946, T: 768, Avg. loss: 17.070864
Total training time: 0.01 seconds.
-- Epoch 49
Norm: 10.23, NNZs: 1, Bias: 1.794972, T: 784, Avg. loss: 17.063910
Total training time: 0.01 seconds.
-- Epoch 50
Norm: 10.24, NNZs: 1, Bias: 1.795102, T: 800, Avg. loss: 17.060152
Total training time: 0.01 seconds.
-- Epoch 51
Norm: 10.24, NNZs: 1, Bias: 1.795160, T: 816, Avg. loss: 17.055316
Total training time: 0.01 seconds.
-- Epoch 52
Norm: 10.24, NNZs: 1, Bias: 1.795090, T: 832, Avg. loss: 17.050635
Total training time: 0.01 seconds.
-- Epoch 53
Norm: 10.25, NNZs: 1, Bias: 1.794892, T: 848, Avg. loss: 17.046109
Total training time: 0.01 seconds.
-- Epoch 54
Norm: 10.25, NNZs: 1, Bias: 1.794834, T: 864, Avg. loss: 17.044928
Total training time: 0.01 seconds.
-- Epoch 55
Norm: 10.25, NNZs: 1, Bias: 1.794763, T: 880, Avg. loss: 17.041549
Total training time: 0.01 seconds.
-- Epoch 56
Norm: 10.25, NNZs: 1, Bias: 1.794559, T: 896, Avg. loss: 17.039110
Total training time: 0.01 seconds.
-- Epoch 57
Norm: 10.26, NNZs: 1, Bias: 1.794203, T: 912, Avg. loss: 17.035666
Total training time: 0.01 seconds.
-- Epoch 58
Norm: 10.26, NNZs: 1, Bias: 1.794083, T: 928, Avg. loss: 17.035048
Total training time: 0.01 seconds.
-- Epoch 59
Norm: 10.26, NNZs: 1, Bias: 1.793933, T: 944, Avg. loss: 17.032884
Total training time: 0.01 seconds.
-- Epoch 60
Norm: 10.26, NNZs: 1, Bias: 1.793557, T: 960, Avg. loss: 17.031787
Total training time: 0.01 seconds.
-- Epoch 61
Norm: 10.26, NNZs: 1, Bias: 1.793166, T: 976, Avg. loss: 17.029581
Total training time: 0.01 seconds.
-- Epoch 62
Norm: 10.26, NNZs: 1, Bias: 1.792895, T: 992, Avg. loss: 17.030173
Total training time: 0.01 seconds.
-- Epoch 63
Norm: 10.27, NNZs: 1, Bias: 1.792659, T: 1008, Avg. loss: 17.028212
Total training time: 0.01 seconds.
-- Epoch 64
Norm: 10.27, NNZs: 1, Bias: 1.792245, T: 1024, Avg. loss: 17.027210
Total training time: 0.01 seconds.
-- Epoch 65
Norm: 10.27, NNZs: 1, Bias: 1.791800, T: 1040, Avg. loss: 17.025539
Total training time: 0.01 seconds.
-- Epoch 66
Norm: 10.27, NNZs: 1, Bias: 1.791443, T: 1056, Avg. loss: 17.026490
Total training time: 0.01 seconds.
-- Epoch 67
Norm: 10.27, NNZs: 1, Bias: 1.791213, T: 1072, Avg. loss: 17.023387
Total training time: 0.01 seconds.
-- Epoch 68
Norm: 10.27, NNZs: 1, Bias: 1.790789, T: 1088, Avg. loss: 17.024484
Total training time: 0.01 seconds.
-- Epoch 69
Norm: 10.27, NNZs: 1, Bias: 1.790246, T: 1104, Avg. loss: 17.021737
Total training time: 0.01 seconds.
-- Epoch 70
Norm: 10.27, NNZs: 1, Bias: 1.789921, T: 1120, Avg. loss: 17.023017
Total training time: 0.01 seconds.
-- Epoch 71
Norm: 10.27, NNZs: 1, Bias: 1.789604, T: 1136, Avg. loss: 17.022113
Total training time: 0.01 seconds.
-- Epoch 72
Norm: 10.27, NNZs: 1, Bias: 1.789135, T: 1152, Avg. loss: 17.021883
Total training time: 0.01 seconds.
-- Epoch 73
Norm: 10.27, NNZs: 1, Bias: 1.788587, T: 1168, Avg. loss: 17.019247
Total training time: 0.01 seconds.
-- Epoch 74
Norm: 10.27, NNZs: 1, Bias: 1.788250, T: 1184, Avg. loss: 17.020409
Total training time: 0.01 seconds.
-- Epoch 75
Norm: 10.28, NNZs: 1, Bias: 1.787875, T: 1200, Avg. loss: 17.020398
Total training time: 0.01 seconds.
-- Epoch 76
Norm: 10.28, NNZs: 1, Bias: 1.787434, T: 1216, Avg. loss: 17.019802
Total training time: 0.01 seconds.
-- Epoch 77
Norm: 10.28, NNZs: 1, Bias: 1.786903, T: 1232, Avg. loss: 17.018054
Total training time: 0.01 seconds.
-- Epoch 78
Norm: 10.28, NNZs: 1, Bias: 1.786466, T: 1248, Avg. loss: 17.019270
Total training time: 0.01 seconds.
-- Epoch 79
Norm: 10.28, NNZs: 1, Bias: 1.786135, T: 1264, Avg. loss: 17.016863
Total training time: 0.01 seconds.
-- Epoch 80
Norm: 10.28, NNZs: 1, Bias: 1.785643, T: 1280, Avg. loss: 17.017837
Total training time: 0.01 seconds.
-- Epoch 81
Norm: 10.28, NNZs: 1, Bias: 1.785090, T: 1296, Avg. loss: 17.016077
Total training time: 0.02 seconds.
-- Epoch 82
Norm: 10.28, NNZs: 1, Bias: 1.784664, T: 1312, Avg. loss: 17.017397
Total training time: 0.02 seconds.
-- Epoch 83
Norm: 10.28, NNZs: 1, Bias: 1.784330, T: 1328, Avg. loss: 17.015116
Total training time: 0.02 seconds.
-- Epoch 84
Norm: 10.28, NNZs: 1, Bias: 1.783802, T: 1344, Avg. loss: 17.015926
Total training time: 0.02 seconds.
Convergence after 84 epochs took 0.02 seconds
</pre>
<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SGDRegressor(eta0=0.001, max_iter=100, random_state=0, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">SGDRegressor</label><div class="sk-toggleable__content"><pre>SGDRegressor(eta0=0.001, max_iter=100, random_state=0, verbose=1)</pre></div></div></div></div></div>



```python
plt.scatter(X_train, y_train, color = 'blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours(train data, SGD)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX4AAAEWCAYAAABhffzLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAsoElEQVR4nO3deXxU1f3/8deHsMgiArLIFlIVaZVWhHzdaC3f4oJ7+7VYFVp+Fr9pXepSW0FRIlpaq9av1rURd3Fh0UqtG0XRuiEgdUHEDQJIhIgiS1C2z++PexMyISGTZG7uTOb9fDzyuHPO3OUzk+QzZ8699xxzd0REJHs0izsAERFpXEr8IiJZRolfRCTLKPGLiGQZJX4RkSyjxC8ikmWU+CUSZnavmf0hRftyM9s3FftqKDN7xcwOSuH+fmBmi1O0r7zwvWqeiv2lIzM7ycweiTuOTKfE3wSY2ffN7FUz+8rMvgiT03/FHVdTY2YnAuvdfUFYvtLMHmzIPt393+7eLyUB1oGZDTGzFSncX0sz+4uZrTCzDWa2xMz+r8o6p5nZHDPbaGarw8fnmJmFz99rZpvNbH34866Z/cnM9ijfh7vPAPqb2fdSFXs2UuLPcGbWHngSuBnoBPQEJgDfpPg4OancXzpKoqX8a+CBOuzPzCxb/scuBfKBg4Hdgf8GFpQ/aWYXAzcB1wF7Ad0I3s/BQMtK+7nW3XcHugBnAocCr5hZ20rrPAwURPZKsoG76yeDfwj+2dbWss7/AouA9cB7wMCw/jvAbGAtsBA4qdI29wK3A08BG4EjgR7AdKAUWAKcv4tj3gvcAcwMj/si0Cd87lbgL1XW/wdwYQ37coIk8SHwZbi9hc81Ay4HioHVwP3AHuFzQ4AVVfa1FDgyfHwlMA14EFgHnEWQuOaF5VXADeG6LYFNQK+wPAzYDGwBNgBvhfWzgYnAK+H6+xIksPL3/xPgV5XiSYgxjO93wNvAV8CjwG41vC85wPXA5+F+zw3fq+bh89UeF2gbxrY9jH1D+Ls9GHgt/HsoAW4BWib5d/jkLn5/e4R/Q6fUso97gT9Uqds9jOW8SnWDgSVx/+9l8k/sAeingb9AaA+sAe4DjgU6Vnl+OPAp8F+AhYmoD9AC+Ai4LExqPwoTRL9wu3vDxDOYILm2AeYD48P19w6TyTE1xHVvuL8jgFYErb2Xw+cOBlYCzcJyZ6AM6FbDvjxMLB2AXIIPnmHhc78MX8feQDvgMeCB8LmEpBrWLSUx8W8Bfhy+xtZh4vt5+Hw74NDw8QHAxir7uhJ4sErdbGBZuH7z8H0+HtgnfP9/GL7WgdXFGMb3BkEi7kSQuH9dw/vya+B9oHe47gskJv6kjxvWDSJoYTcH8sJjX5jk3+Hl4es+B/gu4Qdz+NwwYGt5XLvYx71USfxh/f3Ao5XKncLX2T7u/79M/cmWr6FNlruvA75P8I9wJ1BqZjPMrFu4ylkEX5/neuAjdy8m+AdvB1zj7pvd/XmC5Hp6pd0/4e6vuPt2gn/mLu5+Vbj+J+HxTttFeP9095fc/RtgHHCYmfV29zcIPlSGhuudBsx291W72Nc17r7W3ZcRJLgBYf0Iglb5J+6+gaDL4bQ6nOB8zd3/7u7b3X0TwQfBvmbW2d03uPvr4XodCD7IknGvuy90963uvsXd/+nuH4fv/4vAc8APdrH9X919pbt/QfBNaEAN650K3Ojuy8N1/1T5yboe193nu/vrYdxLgb8RfGAk40/Anwl+H/OAT81sVPhcZ+Bzd99avnJ4TmqtmW0ysyNq2fdKgmRfrvz30CHJ2KQKJf4mwN0Xufv/c/deQH+C1uKN4dO9gY+r2awHsDxM6uWKCc4RlFte6XEfoEf4z7rWzNYSfFvoRs0qtg+T8hfhcSH4hjIyfDyS2vvOP6v0uIzgQ6v8dRRXeQ3Na4mr2hhDo4H9gPfNbK6ZnRDWf0nQ7VDnfZrZsWb2enjifS1wHEEyrElNr7WqHlWOVfl9qPNxzWw/M3vSzD4zs3XAH2uJs4K7b3P3W919MEFCngjcbWbfIfhG2rnyh7G7H+7uHcLnastDPQn+dsqV/x7WJhOb7EyJv4lx9/cJvjL3D6uWE3zdr2ol0LvKycdcgm6hit1VerycoF+1Q6Wf3d39uF2E07v8gZm1I2i1rQyrHgRONrMDCc41/L2211aDlQQfSpVfw1aC/vmNBF1U5THkEJw0rCxheFp3/9DdTwe6ErRgp4UnFj8MdmE9a9q2unoza0VwXuR6gq6sDgTnTSzJ17crJVR6jwlee7LHrS722wm6jvq6e3uCD/Y6x+num9z9VoIPy/0Jus++AU6u677Cv5sjgX9Xqv4OsDT8tiv1oMSf4czs22Z2sZn1Csu9CbpryrsoJgG/M7NB4VUm+5pZH2AOQWK8xMxamNkQ4ESgpmuk3wDWmdkYM2ttZjlm1r+Wy0aPCy81bQlcDcxx9+UA7r4CmEvQ0p8edrPUx8PARWb2rTBJ/JGgP3gr8AGwm5kdb2YtCPqhW+1qZ2Y20sy6hN+E1obV29x9C/AvErs+VgF5tVy50zI8Zimw1cyOBY6u86us3hTgfDPrZWYdgbF1OO4qYM/Kl0oStKTXARvM7NvA2ZUPZmazzezK6gIxswvDS0Rbm1nzsJtnd2CBu68luNLsNjP7qZm1M7NmZjaA4ERzdftrZWaDCBoEXwL3VHr6h8DTNb4rUisl/sy3HjgEmGNmGwkS/rvAxQDuPpXga/dD4bp/Bzq5+2bgJIITwp8DtwG/CL8x7MTdtxF8MAwguKLnc4IPlT2qWz/0EFBI8DV9EEH/b2X3EZw7SPoSyWrcHW7/UhjX18Bvwpi/IjjZOIngm8xGoLZr14cBC81sA8EJ6dPc/evwub8BP6+07tRwucbM3qxuZ+6+HjifIEl/CZwBzKjD69uVO4FngbeANwlObCd13PD3/DDwSdh114PgaqIzCP5O7iS4oqiy3gRXK1VnE/AXgm6qzwmuMDolPBeEu18L/Ba4hODqq1UE7+cY4NVK+7nEzNYT/M3cT3BBweHuvrHSOqeH20o9lV8SJ9LowpN6DwJ5Vc41pC0zexn4jYc3cWWL8BvlVHc/LOY4TiS46urUOOPIdEr8Eouw6+URguvfr4o7HpFsoq4eaXThlR5rge7suPpIRBqJWvwiIllGLX4RkSyTEcO3du7c2fPy8uIOQ0Qko8yfP/9zd69670pmJP68vDzmzZsXdxgiIhnFzIqrq1dXj4hIlok08ZvZBeFkCgvN7MKwrpOZzTSzD8NlxyhjEBGRRJElfjPrTzAO/MHAgcAJZtaX4LbyWe7eF5hF4m3mIiISsShb/N8BXnf3snDclBeBnxAM1HRfuM59BGOhi4hII4ky8b8LHGFme5pZG4IhYXsTjBRYAhAuu1a3sZkVmNk8M5tXWloaYZgiItklssTv7osIhrWdCTxDMJDU1l1ulLh9kbvnu3t+ly47XY0kIiL1FOnJXXe/y90HuvsRBKPtfQisMrPuAOFydZQxiIhIoqiv6ukaLnOB/yEYBnYGUD4l2yjgiShjEBHJRB+s+YBDJx3Kpi31naqiZlHfwDXdzPYkmMf0XHf/0syuAaaY2WiCyZmHRxyDiEjGcHdOnXYq096bBsDclXM5ok9t0xLXTaSJ3913mtjZ3dewY5JtEREJzV85n/w78yvKD/zkgZQnfdCduyIisdvu2xl89+CKpN+tbTfu2edrLj9pJM2aQV4eTJ6cuuNlxFg9IiJN1axPZnHkA0dWlJ8e8TRr5gyjoADKyoK64mIoKAgej6g6gWk9KPGLiMRgy7Yt9L25L8VfBeOoHbTXQcz937nkNMsh78gdSb9cWRmMG6fELyKSkaYunMqp03ZMG/za6Nc4tNehFeVly6rfrqb6ulLiFxFpJBs3b6TjnzuyZfsWAI7vezz/OP0fmFnCerm5QfdOVbm5qYlDJ3dFRBrB7XNvp92f2lUk/YXnLOTJM57cKekDTJwIbdok1rVpE9Snglr8IiIRWlO2hs7Xda4on3XQWdx50p273Ka8H3/cuKB7Jzc3SPqp6N8HJX4RkchMmD2BK1+8sqJcfGExuXsk118zYkTqEn1VSvwiIim2/Kvl5N64I8GPP2I8E/57QowRJVLiFxFJoXP+eQ63z7u9olz6+1I6t+m8iy0anxK/iEgKLCpdxP637V9RvvnYmznv4PNijKhmSvwiIg3g7vzk0Z/wxOJgoGHDWHfpOtq1bBdzZDVT4hcRqac3Pn2DQyYdUlF+5JRH+Fn/n8UYUXKU+EVE6mjb9m0cMukQ5pfMB6B3+958dP5HtMxpGXNkyVHiFxGpg2c/epZhk4dVlJ8b+RxH7XNUjBHVXaSJ38wuAs4CHHgHOBNoAzwK5AFLgVPd/cso4xARaajN2zaTd2MeJRtKADik5yG8OvpVmlnmDYAQWcRm1hM4H8h39/5ADnAaMBaY5e59gVlhWUQkbT3y7iO0+kOriqQ/56w5vH7W6xmZ9CH6rp7mQGsz20LQ0l8JXAoMCZ+/D5gNjIk4DhGROlv/zXraX9O+ovyTb/+E6adOr3Z8nUwS2ceVu38KXE8wr24J8JW7Pwd0c/eScJ0SoGt125tZgZnNM7N5paWlUYUpIlKtm+fcnJD0F527iMd+9ljGJ32IsMVvZh2Bk4FvAWuBqWY2Mtnt3b0IKALIz8/3KGIUEamqdGMpXa/f0R49J/8cbj3+1hgjSr0ou3qOBJa4eymAmT0GHA6sMrPu7l5iZt2B1RHGICKStMufv5yJ/94x9vHyi5bTq32vGCOKRpSJfxlwqJm1ATYBQ4F5wEZgFHBNuHwiwhhERGpVvLaYvJvyKspXDbmKK354RXwBRSyyxO/uc8xsGvAmsBVYQNB10w6YYmajCT4chkcVg4hIbc6acRZ3LbirorzmkjV0at0pxoiiF+lVPe5eCBRWqf6GoPUvIhKbhasX0v/2/hXlO46/g1/l/yrGiBqP7twVkazi7hz30HE889EzALTKacWaS9bQtmXbmCNrPJl594GISBImT4a8PGjWLFgWTnqVZlc1q0j604ZP4+vLv86qpA9q8YtIEzV5MhQUQFkZYNsoPnYgV336NgB7d9yb9899nxY5LeINMiZK/CLSJI0bFyb9Y8+HQ26uqO/6zCw+fu1H8QWWBpT4RaRJKv5sHVy5x46KZYfDPf+mVD3cegdEpOk55sFj4NJKSX/GnXD3K+DNyM2tebtsoRa/iDQZK9atoPf/9U6svHI7EIyv06YNTJy483bZRi1+EWkSet3QKyHpP3XGUzy4r9Onj2EGffpAURGMGBFjkGlCLX4RyWjvrHqH793xvYQ6LwzHdeyrRF8dJX4RyVg2IXGI5PkF8xnYfWBM0WQOJX4RyTjPL3meoffvGPmlfav2fDX2qxgjyizq4xeRne5wnTy5cbevC5tgCUl/yQVLlPTrSIlfJMuV3+FaXAzuwbKgIPnk3dDtkzVm5piErp1ubbvhhU5eh7zUHigLmHv6T26Vn5/v8+bNizsMkSYpLy9I1lX16QNLl0a/fW22bd9G86sTe6VXXLSCnu17NnznTZyZzXf3/Kr1avGLZLlly+pWn+rtd+XYyccmJP0Ou3XAC11Jv4GinHO3H/Bopaq9gfHA/WF9HrAUONXdv4wqDhHZtdzc6lvsyd7h2tDtq7Nx80ba/aldQt36S9fTrmW7GraQuoisxe/ui919gLsPAAYBZcDjwFhglrv3BWaFZRGJycSJwR2tldXlDteGbl9Vt+u7JST9od8aihe6kn4KNdblnEOBj9292MxOBoaE9fcBs4ExjRSHiFRRfoPTuHFB90xubpC0k73xqaHblytZX0KPG3ok1G29Yis5zXLqtiOpVaOc3DWzu4E33f0WM1vr7h0qPfelu3esZpsCoAAgNzd3UHF13yVFpEmoeiPWRYdexA3H3BBTNE1HTSd3I2/xm1lL4CTg0rps5+5FBJOzk5+fn/6XHolInb296m0OvOPAhLqK4RYkMo1xVc+xBK39VWF5lZl1BwiXqxshBhFJoVTcsGUTLCHp33H8HUr6jaQx+vhPBx6uVJ4BjAKuCZdPNEIMIpIiCVMasuOGLUiuX//pD5/muIeOS6hTwm9ckfbxm1kbYDmwt7t/FdbtCUwBcoFlwHB3/2JX+9ENXCLpoyE3bFXty3925LMcvc/RKYtNEsXSx+/uZcCeVerWEFzlIyIZqD43bN029zbOferchDq18uOj0TlFpE7qesNW1Vb+u2e/ywFdD4ggMkmWhmwQkTpJ9oat858+f6ek74WupJ8G1OIXkTqp7Yatrdu30uLqFgnbfHbxZ3Rr162RI5WaKPGLSJ2NGFH9FTxD7h3Ci8UvVpR77t6TFb9d0YiRSTKU+EWkwb7c9CWdru2UULfxso20adGmhi0kTkr8ItIgVfvxT9zvRGacPiOmaCQZSvwiUi8frPmAfrf0S6jToGqZQYlfROqsaiv/pH4n8cRpugk/Uyjxi0jSpi6cyqnTTk2o041YmUeJX0SSUrWVf+aAM7n75LtjikYaQolfRHbp2leuZcy/EudKUis/synxi0iNqrby+eet9Fl9DpP3rfsMW5I+lPhFZCfDpw5n2nvTEiuvDFr5xdRtGGZJP0r8IlLB3Wl2VeIQXt3++Qqr5h6eUFdWFgzZoMSfmZT4RQSAva7fi1UbVyXUeaHTbEL16+9qGGZJb0r8Ilnum63fsNvE3RLqii8sJnePYJzlug7DLOkv0mGZzayDmU0zs/fNbJGZHWZmncxsppl9GC47RhmDSFOVqnlvqyZ9L/SKpA/JD8MsmSPq8fhvAp5x928DBwKLgLHALHfvC8wKyyJSB+Xz3hYXg/uOeW+TTf6rNqza6YqdDZduqPYyzREjoKgomFrRLFgWFal/P5NFNueumbUH3iKYb9cr1S8Ghrh7iZl1B2a7e7+a9gOac1ekqlTOe9u8WXO2XLElZbFJ+qhpzt0oW/x7A6XAPWa2wMwmmVlboJu7lwCEy641BFxgZvPMbF5paWmEYYpknvrMe/tmyZs7Jf1t47cp6WehKBN/c2AgcLu7HwRspA7dOu5e5O757p7fpUuXqGIUyUg1nVjd1by3g4oGVZSP3PvI4Iod0+yr2SjK3/oKYIW7zwnL0wg+CFaFXTyEy9URxiCSUZI9YZvsCddp702rdt7bmT+fmbKYG1sqTmpnu8gSv7t/Biw3s/L++6HAe8AMYFRYNwrQWK4i1O2EbTInXG2CMXzq8IryZd+/LOPH2GnoSW0JRHZyF8DMBgCTgJbAJ8CZBB82U4BcYBkw3N2/2NV+dHJXskFDTthWVvhCIVe9dFVCXaYn/HKpeo+yRU0ndyO9gcvd/wPsdFCC1r+IVFKfE7ZVVe3WeeSUR/hZ/581IKr0kor3SHTnrkjaaMgdssdOPpZnPnomoa6ptPIr013EqaFT+iJpoj53yLo7NsESkv68/53XJJM+6C7iVFGLXyRNlJ+YHTcu6LrIzQ0SWk13yPa7pR8frPkgoa6pJvxydX2PpHqRntxNFZ3cFdlh05ZNtPljYrN35W9X0n337jFFJOkqlpO7IpJaO82IRdNv5UvqKfGLZICS9SX0uKFHQt3X476mVfNWMUUkmUyJXyTNVW3lf7frd3n77LdjikaaAiV+kTS1oGQBA4sGJtRtH78ds527e0TqQolfJA1VbeWPPmg0k06aFFM00tQo8YukkccWPcYpU05JqNPJW0k1JX6RNFG1lf/XYX/lN4f8JqZopClT4heJ2R9e+gNXvHBFQp1a+RIlJX6RGFVt5T878lmO3ufomKKRbJF04jez1kCuuy+OMB6RrHDKlFN4bNFjCXVq5UtjSSrxm9mJwPUE4+p/Kxxn/yp3PynC2ESanO2+nZyrchLqFp27iG93/nZMEUk2SrbFfyVwMDAbgnH2zSyvto3MbCmwHtgGbHX3fDPrBDwK5AFLgVPd/cu6hS2Sebr/pTufbfgsoU6tfIlDssMyb3X3r+p5jP929wGVBgoaC8xy977ALOowAbtIJtqweQM2wRKS/ppL1ijpS2ySbfG/a2ZnADlm1hc4H3i1nsc8GRgSPr6P4FvEmHruSyStaVA1SUfJtvh/AxwAfAM8BHwFXJjEdg48Z2bzzawgrOvm7iUA4bJrdRuaWYGZzTOzeaWlpUmGKZIeln+1fKekv/nyzUr6khZqbfGbWQ4ww92PBMbVcf+D3X2lmXUFZprZ+8lu6O5FQBEE4/HX8bgisama8A/rdRivjq7vF2SR1Ku1xe/u24AyM9ujrjt395XhcjXwOMEJ4lVm1h0gXK6u635F4jJ5MuTlQbNmwXLy5B3PzVkxZ6ekv338diV9STvJ9vF/DbxjZjOBjeWV7n5+TRuYWVugmbuvDx8fDVwFzABGAdeEyyfqGbtIo5o8GQoKoKwsKBcXB2WAkR8lJvzzDz6fm469qZEjFElOson/n+FPXXQDHg+HkG0OPOTuz5jZXGCKmY0GlgHD67hfkViMG7cj6Zcr2+chRn6UOOGr+vEl3SWV+N39PjNrCewXVi129y21bPMJcGA19WuAoXUNVCRuy5ZVqbgysZV/54l3ctbAsxovIJF6SvbO3SEEl14uBQzobWaj3P2lyCITSTO5uUH3Dv91Gxx/bsJzauVLJkm2q+cvwNHl4/SY2X7Aw8CgqAITSTcTJ+7cl9/q4Re464oh8QQkUk/JXsffovLgbO7+AdAimpBE0s/vn/v9Tkm/zz3OXVcMYcSIGjYSSVPJtvjnmdldwANheQQwP5qQRNLHtu3baH514r/JiotW0LN9TyiMKSiRBko28Z8NnEswVIMBLwG3RRWUSDoY9uAwnv342Ypyx9068sWYL2KMSCQ1kk38zYGb3P0GqLibt1VkUYnEaOPmjbT7U7uEuvWXrqddy3Y1bCGSWZLt458FtK5Ubg38K/XhiMSry3VdEpL+UXsfhRe6kr40Kcm2+Hdz9w3lBXffYGZtIopJpNGtXL+Snjf0TKjbesVWcprl1LCFSOZKtsW/0cwGlhfMLB/YFE1IIo3LJlhC0r/o0IvwQlfSlyYr2Rb/BcBUM1tJMNRyD+BnkUUl0gje+uwtBvxtQEKdbsSSbJBs4v8WcBCQC/wEOJTgA0AkI1UdRfNvJ/yNgkEFNawt0rQk29VzhbuvAzoARxGMk397VEGJpFLloZS7ff+pnZK+F7qSvmSVZFv828Ll8cAd7v6EmV0ZTUgiqZMwlPKVljD5w3Mjn+OofY6KKzSR2CTb4v/UzP4GnAo8ZWat6rCtSGzGjYOyHs/sNJJmn3tcSV+yVrIt/lOBYcD17r42nDnr99GFJdJw7k7xmVXaJ7e+C6UHsGznOdBFskZSrXZ3L3P3x9z9w7Bc4u7PJbOtmeWY2QIzezIsdzKzmWb2YbjsWP/wRap394K7aXZVpT/vT4bClQ6lBwDBEMsi2SrZFn9DXAAsAtqH5bHALHe/xszGhuUxjRCHZIHqBlVrfeNaNq3dMWV0mzbBEMsi2SrSfnoz60VwQnhSpeqTCSZ1IVz+OMoYJHuMf2F8QtI/O/9svNC585Y96NMHzKBPHygqQkMpS1aLusV/I3AJsHulum7uXgJBl5GZda1uQzMrAAoAcvW9XHZh05ZNtPlj4ggi31z+DS1zWgJBkleiF9khsha/mZ0ArHb3eo3b7+5F7p7v7vldunRJcXTSVIx8bGRC0r/2yGvxQq9I+iKysyhb/IOBk8zsOGA3oL2ZPQisMrPuYWu/OyRcWi2SlM/LPqfLdYkNgu3jt2Omy3VEahNZi9/dL3X3Xu6eB5wGPO/uI4EZwKhwtVHAE1HFIE1TflF+QtJ/+JSH8UJX0hdJUmNc1VPVNcAUMxsNLAOGxxCDZKCPv/iYfW/eN6FOg6qJ1F2jJH53nw3MDh+vAYY2xnGl6Wg9sTVfb/26ojx71Gx+mPfDGCMSyVxxtPhFkvbGp29wyKRDEurUyhdpGCV+SVtVR9FceM5C9u+yf0zRiDQdGmhN0s4/Fv8jIenv3XFvvNCV9EVSRC1+SRvunji+DvDpbz+lx+49YopIpGlSi1/Swu1zb09I+sf3PR4vdCV9kQioxS+x2rp9Ky2ubpFQt27sOnZvtXsNW4hIQ6nFL7EZ+6+xCUn/gkMuwAtdSV8kYmrxS6PbuHkj7f7ULqFu8+WbaZHTooYtRCSV1OKXRjV86vCEpH/jMTfiha6kL9KI1OKXRrFqwyr2+steCXUaVE0kHmrxS+T639Y/IelPP3W6BlUTiZFa/BKZD9Z8QL9b+iXUabgFkfgp8Uskqg638MovX+Hw3ofHFI2IVKbELyn16vJXGXz34IQ6tfJF0osSv6RM1Vb+4vMWs9+e+8UUjYjURCd3pcGmvzc9Iekf0OUAvNCV9EXSVJSTre9mZm+Y2VtmttDMJoT1ncxsppl9GC47RhWDRMvdsQnGT6f+tKLus4s/491z3q1128mTIS8PmjULlpMnRxeniCSKssX/DfAjdz8QGAAMM7NDgbHALHfvC8wKy5Jhbnr9poRB1U75zil4odOtXbdat508GQoKoLgY3INlQYGSv0hjiayP390d2BAWW4Q/DpwMDAnr7yOYknFMVHFIam3ZtoWWf2iZULfh0g20bdk26X2MGwdlZYl1ZWVB/YgRqYhSRHYl0j5+M8sxs/8Aq4GZ7j4H6ObuJQDhsmsN2xaY2Twzm1daWhplmJKki565KCHpjxk8Bi/0OiV9gGXL6lYvIqkV6VU97r4NGGBmHYDHzax/HbYtAooA8vPzdT1gjNZ/s57217RPqNtyxRaaN6vfn09ubtC9U129iESvUa7qcfe1BF06w4BVZtYdIFyubowYpH5OeOiEhKR/23G34YVe76QPMHEitGmTWNemTVAvItGLrMVvZl2ALe6+1sxaA0cCfwZmAKOAa8LlE1HFIPW3cv1Ket7QM6EuVYOqlffjjxsXdO/k5gZJX/37Io0jyq6e7sB9ZpZD8M1iirs/aWavAVPMbDSwDBgeYQxSD/v8dR8++fKTivKM02ZwYr8TU3qMESOU6EXiEuVVPW8DB1VTvwYYGtVxpf7eK32PA247IKFOwy2IND26czeLVb6JyiZYQtKfc9YcJX2RJkpj9WSp8puoyrq8CIVDKupb2G5sHr8pvsBEJHJK/Flq3Dgou6TKidqbPqJH+31gfDwxiUjjUFdPFnr4nYcpPrNS0l85CK50+HIf3UQlkgXU4s8i7p4wvg4A15ZCWeeKom6iEmn61OLPEte9cl1C0j+83QjaXOsJSV83UYlkB7X4m7jN2zbT6g+tEurKLiujdYvWTN5LN1GJZCO1+Juws588OyHpjz9iPF7otG7RGgiS/NKlsH17sFTSF8kOavE3QWu/XkvHPyfOb7P1iq3kNMuJKSIRSSdq8TcxQ+8fmpD0J504CS90JX0RqaDE30Qs/2o5NsF4fsnzFXXbx29n9MDR9d6npkcUaZrU1dME9PhLD0o2lFSUnx7xNMP2HdagfVbc2RvOlFU+PSLoXIBIprNghsT0lp+f7/PmzYs7jLTz9qq3OfCOAxPqUjW+Tl5e9ZOl9OkTnAgWkfRnZvPdPb9qvVr8GcomJA63ML9gPgO7D0zZ/jU9okjTpT7+DPOvT/6VkPQ7te6EF3pKkz7UfAev7uwVyXyRJX4z621mL5jZIjNbaGYXhPWdzGymmX0YLjvWti8J2ATjqAeOqigvvWApay5ZE8mxND2iSNMVZYt/K3Cxu38HOBQ418z2B8YCs9y9LzArLMsuPPfxcwmt/MG9B+OFTp8OfSI75ogRUFQU9OmbBcuiIp3YFWkKopyBqwQoCR+vN7NFQE/gZGBIuNp9BJOwj4kqjkzm7gy9fygvLH2hou6LS76gY+vG+ZKk6RFFmqZGOblrZnkE0zDOAbqFHwq4e4mZda1hmwKgACA3CzuWXyp+iR/e+8OK8pSfTmH4AZqeWEQaLvLEb2btgOnAhe6+zsxq2wQAdy8CiiC4nDO6CNPL1u1b6X9bfxavWQzAfnvux8JzFtK8mS7AEpHUiPSqHjNrQZD0J7v7Y2H1KjPrHj7fHVgdZQyZ5O/v/50WV7eoSPqzR81m8XmLlfRFJKUiyygWNO3vAha5+w2VnpoBjAKuCZdPRBVDpti0ZRNdr+/Khs0bABiSN4Tnf/E8yX47EhGpiyibkoOBnwPvmNl/wrrLCBL+FDMbDSwDsrrj+u4FdzN6xo7xdBb8agED9hoQX0Ai0uRFeVXPy0BNTdahUR03U1QdOvmM757B5P/RKGgiEj11Hsfgzy//mbGzdty+8PH5H7N3x71jjEhEsokSfyNauX4lPW/oWVH+3WG/47qjr4sxIhHJRkr8jeSiZy7ixjk3VpRLLi5hr3Z7xReQiGQtJf6IfbjmQ/a7Zb+K8nVHXcfvDv9djBGJSLZT4o+Iu3P69NN5dOGjFXVrx6xlj932iDEqEREl/ki8WfImg4oGVZTv+/F9/OLAX8QYkYjIDhqPP4W2+3Z+cM8PKpL+nq33ZNO4TQ1K+pr3VkRSTS3+FHlhyQv86P4fVZSfPP1Jjt/v+AbtU/PeikgUNOduA23ZtoV+t/RjydolAHy363dZ8KsF5DTLafC+Ne+tiDSE5tyNwPT3pvPTqT+tKL985ssMzh2csv1r3lsRiYISfz2UbSmj05878c22bwA4Zp9jeHrE0ykfVC03t/oWfxZOTyAiKaSTu3VUNL+Itn9sW5H03/712zwz8plIRtLUvLciEgW1+JP0xaYv2PPaPSvKvxzwS+46+a5Ij1l+AnfcuKB7Jzc3SPo6sSsiDaHEn4SrX7ya8bPHV5SXXLCEvA55jXJszXsrIqmmxL8Ln677lF7/16uifNn3L2PiUPWziEhmi6yP38zuNrPVZvZupbpOZjbTzD4Mlx13tY+GaOiNT+c9dV5C0l/9u9VK+iLSJER5cvdeYFiVurHALHfvC8wKyylXfuNTcTG477jxKZnkv/jzxdgE49a5twJw4zE34oVOl7ZdoghVRKTRRXoDl5nlAU+6e/+wvBgY4u4l4UTrs929X237qesNXPW58cndOWXKKTz+/uMVdevGrmP3VrsnfVwRkXSSLjdwdXP3EoAw+XetaUUzKwAKAHLreOF6XW98mvvpXA6edHBFefL/TOaM755Rp2OKiGSKtD256+5FQBEELf66bJvsjU/bfTuH3XUYb3z6BgDd23VnyQVLaNW8Vf2CFhHJAI19A9eqsIuHcLk6ioMkc+PTzI9nknNVTkXSf3rE06y8eKWSvog0eY3d4p8BjAKuCZdPRHGQXd34tHnbZvb56z6sWLcCgEHdBzHnrDkpGVRNRCQTRHZy18weBoYAnYFVQCHwd2AKkAssA4a7+xe17StVo3M++u6jnDb9tIrya6Nf49BehzZ4vyIi6ajRT+66++k1PDU0qmPWZMPmDexxzR5s9+0AnLjfiTxx2hORjK8jIpLu0vbkbqrc+satnPf0eRXlhecsZP8u+8cYkYhIvJr06Jx3vXlXRdIvGFiAF7qSvohkvSbd4u/ftT+H9z6cR055hN579I47HBGRtNCkE/8hvQ7hlV++EncYIiJppUl39YiIyM6aVOLPhInjRUTi1qQSv4iI1K5JJX5dly8iUrsmlfhFRKR2SvwiIllGiV9EJMso8YuIZJlIp15MFTMrBaqZWqXRdQY+jzuIGqRrbOkaFyi2+kjXuCB9Y4szrj7uvtOE4RmR+NOFmc2rbojTdJCusaVrXKDY6iNd44L0jS0d41JXj4hIllHiFxHJMkr8dVMUdwC7kK6xpWtcoNjqI13jgvSNLe3iUh+/iEiWUYtfRCTLKPGLiGQZJf4kmNndZrbazN6NO5aqzKy3mb1gZovMbKGZXRB3TABmtpuZvWFmb4VxTYg7psrMLMfMFpjZk3HHUpmZLTWzd8zsP2Y2L+54KjOzDmY2zczeD//eDkuDmPqF71X5zzozuzDuuMqZ2UXh3/+7Zvawme0Wd0ygPv6kmNkRwAbgfnfvH3c8lZlZd6C7u79pZrsD84Efu/t7McdlQFt332BmLYCXgQvc/fU44ypnZr8F8oH27n5C3PGUM7OlQL67p92NSGZ2H/Bvd59kZi2BNu6+NuawKphZDvApcIi7x37Dp5n1JPi739/dN5nZFOApd7833sjU4k+Ku78EfBF3HNVx9xJ3fzN8vB5YBPSMNyrwwIaw2CL8SYtWhpn1Ao4HJsUdS6Yws/bAEcBdAO6+OZ2Sfmgo8HE6JP1KmgOtzaw50AZYGXM8gBJ/k2JmecBBwJyYQwEqulP+A6wGZrp7WsQF3AhcAmyPOY7qOPCcmc03s4K4g6lkb6AUuCfsIptkZm3jDqqK04CH4w6inLt/ClwPLANKgK/c/bl4owoo8TcRZtYOmA5c6O7r4o4HwN23ufsAoBdwsJnF3k1mZicAq919ftyx1GCwuw8EjgXODbsZ00FzYCBwu7sfBGwExsYb0g5h19NJwNS4YylnZh2Bk4FvAT2AtmY2Mt6oAkr8TUDYhz4dmOzuj8UdT1Vhl8BsYFi8kQAwGDgp7Et/BPiRmT0Yb0g7uPvKcLkaeBw4ON6IKqwAVlT61jaN4IMgXRwLvOnuq+IOpJIjgSXuXuruW4DHgMNjjglQ4s944UnUu4BF7n5D3PGUM7MuZtYhfNya4J/g/ViDAtz9Unfv5e55BF0Dz7t7WrTCzKxteIKesBvlaCAtriRz98+A5WbWL6waCsR6AUEVp5NG3TyhZcChZtYm/D8dSnAOLnZK/Ekws4eB14B+ZrbCzEbHHVMlg4GfE7Rcyy9pOy7uoIDuwAtm9jYwl6CPP60unUxD3YCXzewt4A3gn+7+TMwxVfYbYHL4Ox0A/DHecAJm1gY4iqBFnTbCb0fTgDeBdwjybVoM36DLOUVEsoxa/CIiWUaJX0Qkyyjxi4hkGSV+EZEso8QvIpJllPhFQmaWl44jsIqkmhK/SITCwblE0ooSv0iiHDO7MxxD/Tkza21mA8zsdTN728weD8dgwcxmm1l++LhzOAwEZvb/zGyqmf2DYMC17mb2Unhz3btm9oP4Xp6IEr9IVX2BW939AGAtcApwPzDG3b9HcAdmYRL7OQwY5e4/As4Ang0HrDsQ+E/qwxZJnr6GiiRa4u7/CR/PB/YBOrj7i2HdfSQ3AuRMdy+fw2EucHc4mN7fK+1fJBZq8Ysk+qbS421Ah12su5Ud/0NVp9TbWP4gnMjnCILZoR4ws180PEyR+lPiF9m1r4AvK/XL/xwob/0vBQaFj39a0w7MrA/BHAB3Eoykmk7DGUsWUlePSO1GAXeEo0B+ApwZ1l8PTDGznwPP72L7IcDvzWwLwdzNavFLrDQ6p4hIllFXj4hIllHiFxHJMkr8IiJZRolfRCTLKPGLiGQZJX4RkSyjxC8ikmX+PxyP8bpn7CcwAAAAAElFTkSuQmCC"/>
