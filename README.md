# 本地生化反应网络仿真服务

一个本地运行的确定性生化反应网络时间历程仿真服务（FastAPI + NumPy）。
同一网络在**重复执行、改变输出步长、分段求值、并发请求、参数扰动与批量扫描**下，
给出一致、可复现、可解释的结果；非法输入按原因区分、显式报错，绝不静默修正。

## 1. 快速开始

```bash
uv sync
uv run pytest -q          # 运行全部测试（34 个）
uv run python main.py     # 启动服务 http://127.0.0.1:8000
```

### 端到端示例

```bash
curl -X POST http://127.0.0.1:8000/networks -H 'Content-Type: application/json' -d '{
  "network": {
    "model_id": "ab", "version": 1,
    "species": [
      {"id": "A", "initial_concentration": 2.0, "composition": {"X": 1}},
      {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}}
    ],
    "reactions": [{
      "id": "r", "reactants": {"A": 1}, "products": {"B": 1},
      "reversible": true,
      "forward": {"kind": "mass_action", "k": 1.0},
      "reverse": {"kind": "mass_action", "k": 3.0}
    }]
  }
}'

curl -X POST http://127.0.0.1:8000/simulate -H 'Content-Type: application/json' -d '{
  "model_id": "ab", "config": {"t_end": 30.0, "output_step": 1.0}
}'

curl -X POST http://127.0.0.1:8000/scan -H 'Content-Type: application/json' -d '{
  "model_id": "ab",
  "config": {"t_end": 30.0, "output_step": 1.0},
  "parameter_path": "reactions.r.forward.k",
  "values": [0.5, 1.0, 2.0]
}'
```

## 2. 网络描述约定

```jsonc
{
  "model_id": "ab",          // 非空字符串
  "version": 1,              // 正整数；相同 (model_id, version) 内容必须一致
  "species": [
    {"id": "A",
     "initial_concentration": 2.0,   // 必须 >= 0 的有限数
     "composition": {"X": 1}}        // 可选：原子组成，原子量必须为正
  ],
  "reactions": [{
    "id": "r",
    "reactants": {"A": 1},           // 计量数：有限正数（0 非法，应省略物种）
    "products":  {"B": 1},
    "reversible": true,              // 可省略，默认 false
    "forward": {"kind": "mass_action", "k": 1.0},
    "reverse": {"kind": "mass_action", "k": 3.0}  // reversible=true 时必填
  }]
}
```

### 速率定律（rate law）

| kind | 参数 | 速率 | 约束 |
|---|---|---|---|
| `mass_action` | `k >= 0` | `k · Π [S_i]^ν_i` | 通用，支持多反应物 |
| `enzymatic`（Michaelis–Menten） | `k >= 0`（即 k_cat）、`km > 0`、`enzyme` | `k_cat · [E] · [S] / (Km + [S])` | 该方向必须恰好有一个底物；`enzyme` 必须是已声明物种且不化学计量参与反应；酶初始浓度必须 > 0 |

可逆反应展开为两条有向反应（正向、反向），反向速率以产物为底物按相同规则求值。

### 闭合性（守恒）约定

* 每个物种可用 `composition` 声明原子组成（如葡萄糖 `{"C":6,"H":12,"O":6}`）。
* **未声明 `composition` 的物种视为由与其同名的单位原子构成。** 这不会"自动修正"网络：
  凭空生成该物种的反应仍会因原子不平衡而被拒绝。
* 每条反应都必须对每一种原子满足 `Σ 产物计量×组成 == Σ 反应物计量×组成`，
  否则以 `unclosed_network` 拒绝（恒化源/汇不允许出现在闭合网络中）。
* 守恒总量按原子组成矩阵逐种原子计算；酶等催化剂不计入计量变化，浓度自然守恒。

### 参数扰动路径（override / scan）

* `species.<id>.initial_concentration`
* `reactions.<id>.forward.k` / `.km`，`reactions.<id>.reverse.k` / `.km`

## 3. 数值方法、容差与收敛判定

* **积分器**：经典定步长 RK4；在每个输出区间内独立积分，速率无显式时间依赖，
  因此把 `[0, t_end]` 分段（断点必须落在输出网格上）与一次性积分产生
  **逐位相同**的轨迹（测试中断点比较差异为 `0.0`）。
* **内部步长收敛**：对每个输出区间做理查森加密（`n` 与 `2n` 个 RK4 子步比较），
  误差判据为逐分量 `|y_2n - y_n| <= abs_tol + rel_tol·max(|y_n|,|y_2n|)`，
  不满足则继续二分，最多 12 次；仍不收敛抛 `integration_failed`，绝不返回不准结果。
* **输出步长无关性**：调小 `output_step` 只是对已收敛轨迹重新采样，
  不改变稳态与轨迹结论（测试比较共享时刻，差异 < 1e-8）。
* **守恒容差**：`conservation_tol`（默认 `1e-6`，相对值），
  逐输出点检查每种原子总量相对初始总量的偏差；越界抛 `conservation_violation`。
* **非物理防护**：`negative_conc_tol`（默认 `1e-10`）。
  负于该容差立即抛 `non_physical_result`（携带时间点、物种索引与浓度）；
  容差内的浮点噪声钳制为 0。
* **稳态判定**：终点处 `max_i |dy_i/dt| / max(|y_i|, 1) <= 1e-8` 时，
  在结果 `steady_state` 中返回各物种浓度，否则为 `null`。
* 配置默认值：`rel_tol=1e-6`、`abs_tol=1e-9`。`t_end`、`output_step` 及所有容差必须 > 0。

## 4. 错误分类

所有错误返回 `{"error": {"code", "message", "details"}}`，`code` 稳定可区分：

| code | 含义 | 典型 details |
|---|---|---|
| `invalid_initial_state` | 初始浓度缺失/为负/非有限 | `species`, `initial_concentration` |
| `missing_species` | 反应（或酶）引用了未声明物种 | `species`, `declared` |
| `invalid_stoichiometry` | 计量数非数/为负/为 0、反应空、净变化为 0 | `reaction`, `species`, `coefficient` |
| `invalid_rate` | 速率定律缺失、k<0、缺 km/km<=0、可逆缺反向定律、酶底物不合法、酶初值为 0 | `reaction`, `direction`, `enzyme` |
| `unclosed_network` | 反应原子不平衡（网络不闭合） | `reaction`, `imbalance` |
| `invalid_config` | 仿真配置缺字段或不可行 | 对应字段名 |
| `integration_failed` | 加密后仍达不到容差 / 断点不对齐网格 | `sub_steps`, `max_error` |
| `conservation_violation` | 守恒总量漂移越界 | `max_relative_error`, `tolerance` |
| `non_physical_result` | 出现非物理负浓度 | `time`, `species_index`, `concentration` |
| `unknown_model` | 模型 id/版本未注册（HTTP 404） | `model_id`, `available` |

校验按固定顺序执行（初始状态 → 缺失物种 → 计量 → 速率 → 闭合性），首个原因确定且可复现。
**单次接口与批量扫描共用同一个 `validate_network`**：扫描中某个扰动值非法时，
该扫描点返回相同 `error_code`，其他点正常返回，不会被跳过或静默修正。

## 5. 一致性与隔离保证

* 注册的网络是不可变 frozen 对象，保存在只增注册表中；相同 `(model_id, version)`
  重复注册但内容不同会被拒绝（防止版本内容被悄悄替换）。
* 每次运行/每个扫描点都通过 `dataclasses.replace` 构造**独立的网络副本**施加扰动，
  不在请求、扫描点、线程间共享任何可变状态。
* 不做跨参数/跨版本的结果缓存——结果始终由确定的输入重新计算，
  因此不同模型版本或速率参数绝不可能被错误复用；相同请求天然可逐位复现。

## 6. API

| 方法/路径 | 说明 |
|---|---|
| `POST /networks` | 导入并校验网络，返回 `model_id/version` |
| `POST /simulate` | `{model_id, version?, config, overrides?}` 时间历程仿真 |
| `POST /scan` | `{model_id, version?, config, parameter_path, values[]}` 批量参数扫描，逐点返回 `ok/result/error_code` |
| `GET /` | 健康检查 |

`version` 省略时使用该 model 的最高版本。

## 7. 验证方法

```bash
uv run pytest -q -s          # -s 查看关键输入与判定依据日志
```

测试覆盖（`tests/`）：

* `test_validation.py`：缺失物种/酶、非法计量、零净反应、速率缺失/不可行、
  可逆反应缺反向定律、酶初值为 0、未闭合网络、源反应、负初值、校验顺序确定性。
* `test_simulation.py`：闭合网络守恒（单原子/多原子）、酶促反应催化剂量恒定、
  缩小输出步长结论不变、分段求值与单次积分逐点相同、守恒失效被检出、
  负浓度被检出并定位时间点、容差内噪声钳制、非法配置拒绝。
* `test_isolation.py`：单次扰动不污染已注册模型、扫描点相互独立、
  非法扫描值就地报错、8 线程并发扰动与 3 路并发扫描隔离、
  不同模型版本结果不被混用、同版本内容冲突被拒。
* `test_api.py`：HTTP 错误码可区分且可定位、单接口与扫描对同一网络的校验结论一致、
  扫描返回全部点、未知模型 404。

日志在每次运行打印 `model_id/version/overrides/t_end/step` 与
`守恒漂移/最小浓度/是否达到稳态`；失败日志打印 code、输入载荷与 details，便于定位。
