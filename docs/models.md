## 定式化一覧

### カットセット定式化

### フロー

### MTZ

$$
\begin{aligned}
\min\quad \sum C_{ij}x_{ij} \\
s.t. \quad \sum_{j}x_{ij} &= 1, \quad i\neq 0\\
\sum_{j}x_{ji} &= 1, \quad i\neq 0\\
\quad \sum_{j}x_{0j} &= m,\\
\sum_{j}x_{j0} &= m,\\
u_i + q_j - Q(1 - x_{ij}) &\leq u_{j}, \quad j \neq 0\\
q_i \leq u_i &\leq Q,\\
x_i &\in \{0, 1\}.
\end{aligned}
$$

### MTZ2

$$
\begin{aligned}
\min\quad \sum_{aij} C_{ij}x^a_{ij} \\
s.t. \quad \sum_{aj}x^a_{ij} &= 1,\\
\sum_{aj}x^a_{ji} &= 1,\\
u_i + q_j - Q(1 - \sum_{a}x^a_{ij}) &\leq u_{j}, \quad j \neq 0\\
q_i \leq u_i &\leq Q,\\
x_i &\in \{0, 1\}.
\end{aligned}
$$

### 3添字フロー


### 3添字カットセット

$$
\begin{aligned}
\min\quad \sum_{aij} C_{ij}x^a_{ij} \\
s.t. \quad \sum_{j}x^a_{ij} &= y^{a}_{i},\quad i\neq0\\
\sum_{j}x^a_{ji} &= y^{a}_{i},\quad i\neq0\\
\sum_{i}x^{a}_{0i} &= 1,\\
\sum_{i}x^{a}_{i0} &= 1,\\
\sum_{i}q_{i}y^{a}_{i} &\leq Q,\\
\sum_{(i,j)\in S}x_{ij}&\leq |S| - 1,\\
q_i \leq u_i &\leq Q,\\
x_i &\in \{0, 1\}.
\end{aligned}
$$

### 列生成法

集合分割問題としての定式化

$$
\begin{aligned}
\min\quad \sum C_{r}z_{r},\\
s.t.\quad \sum_{r} A_{ir}z_{r} &\geq 1,\\
\sum_{r}z_{r} &= M,\\
z_{r}&\in \{0, 1\}
\end{aligned}
$$

LP緩和の双対問題

$$
\begin{aligned}
\max\quad \sum_{i\neq 0} y_{i} + My_{0},\\
s.t.\quad y_{0} + \sum_{i}A_{ir}y_{i} &\leq C_{r},\\
y_{i} &\geq 0. \quad i \neq 0
\end{aligned}
$$


スレーブ問題では、制約 $C_r - My^\ast_{0} - \sum_{i} A_{ir}y^\ast_{i} \geq 0$ を破るルートを見つける。

$$
\begin{aligned}
\min\quad \sum_{ij} C_{ij}x_{ij} - \sum_{i}y^\ast_{i}w_{i} -  My^\ast_{0},\\
s.t.\quad \sum_{j}x_{ij} &= w_{i},\quad i\neq 0\\
\sum_{j}x_{ji} &= w_{i},\quad i\neq 0\\
\sum_{j}x_{0j} &= 1,\\
\sum_{j}x_{j0} &= 1,\\
\sum_{i}Q_{i}w_{i} &\leq Q, \\
\sum_{(i,j)\in S_0}x_{ij} &\leq |S_0| - 1,\\
x_{ij} &\in \{0, 1\},\\
w_{i} &\in \{0, 1\}.
\end{aligned}
$$

1. まず適当なルートを初期解として導入する
    - 例えば、1箇所に行って帰ってくるルート1つ