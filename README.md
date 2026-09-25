# 本地生化反应网络仿真服务

对本地生化反应网络做导入、校验、确定性时间历程仿真、参数扰动与并发批量扫描。
重复执行、改变输出分块、分段求值与并发请求下结果一致且可解释。

## 运行

```bash
uv sync
uv run python main.py          # 启动 HTTP 服务 (127.0.0.1:8000)
uv run pytest                  # 全部测试，日志打印关键输入与判定依据
```

## 网络描述约定

网络为不可变数据结构（`app/models.py`），JSON 请求体示例：

```json
{
  "version": "model-1",
  "species": {"A": 1.0, "B": 0.0},
  "boundary_species": [],
  "reactions": [
    {
      "name": "r1",
      "reactants": {"A": 1},
      "products": {"B": 1},
      "rate":     {"kind": "mass_action", "params": {"k": 2.0}},
      "reverse":  {"kind": "mass_action", "params": {"k": 1.0}}
    }
  ]
}
```

- **物种**：`species` 给出名称与非负有限初始浓度；必须全部显式声明，不允许隐式创建。
- **反应**：`reactants` / `products` 为 `物种 -> 化学计量系数`，系数必须为有限正数；
  可逆反应通过 `reverse` 速率律表达，其反应物为正反应的产物。
- **速率律**：
  - `mass_action`：参数 `k >= 0`，速率 `v = k·∏[反应物]^系数`；
  - `michaelis_menten`：参数 `kcat >= 0`、`Km >= 0`、`enzyme`（物种名字符串，
    不被消耗），速率 `v = kcat·[E]·[S]/(Km+[S])`，限定单底物。
- **闭合/开放网络**：`boundary_species` 为空即闭合网络，此时禁止源反应（`∅→X`）
  与汇反应（`X→∅`）；边界物种浓度恒定，开放网络允许源/汇。

## 错误分类

错误**不会被静默修正或吞掉**，统一以 `422 {"code", "detail"}` 返回（离线调用抛
`NetworkValidationError` / `SimulationError`），不同原因代码可区分：

| code | 触发条件 |
| --- | --- |
| `MISSING_SPECIES` | 反应或速率律引用了未声明物种（含酶），或网络无物种、边界物种未声明 |
| `ILLEGAL_STOICHIOMETRY` | 计量系数非正/非有限，初始浓度为负/非有限，反应两侧皆空 |
| `INFEASIBLE_RATE` | 速率律类型未知、必需参数缺失、参数为负或非有限，或 ODE 求解失败 |
| `UNCLOSED_NETWORK` | 闭合网络中存在源或汇反应 |
| `CONSERVATION_VIOLATION` | 闭合网络总物质量偏离初值超过容差 |
| `NEGATIVE_CONCENTRATION` | 浓度低于 `-1e-9`（非物理负值） |
| `DUPLICATE_SPECIES` | 输入中存在重复边界物种条目 |
| `UNKNOWN_PARAMETER` | 扰动/扫描指定了不存在的反应或参数 |

`/validate`、`/simulate`、`/scan` 与离线批量扫描共用同一个
`validate_network()`，同一网络校验结论必然一致。

## 数值方法、容差与收敛判定

- ODE 用 LSODA（自适应刚性/非刚性），固定 `rtol=1e-9, atol=1e-12`，从 t=0
  积分一次并保留**稠密解**。输出网格（`n_points`）和时间分段只对同一稠密解
  采样，因此调小输出步长、把区间分段求值都不会改变轨迹与稳态结论。
- **守恒判定**：闭合网络对全部自由物种逐时刻求和，最大偏差
  `max|Σ(t)−Σ(0)| ≤ conservation_tol · max(|Σ(0)|,1)`，默认容差 `1e-6`；
  超限抛 `CONSERVATION_VIOLATION`，错误信息含实际偏差与容差。
- **非负判定**：求解器噪声范围内（≥ `-1e-9`）的微小负值在输出端钳为 0；
  低于该阈值抛 `NEGATIVE_CONCENTRATION`。
- **稳态结论**：在足够长的时间区间（如 20 个时间单位）末端读数；
  时间历程测试对解析平衡点（如可逆反应 kf/kr=2 时 [A]=1/3, [B]=2/3）校验。

## 隔离、复现与缓存键

- 网络与反应均为 frozen 数据结构；参数扰动（`perturb_parameter`）总是返回**新**
  网络，原对象不变。
- 每次扫描对每个参数值独立构建网络并仿真，无线程间可变共享；
  集成解缓存以网络全量描述（物种、计量、速率参数、边界集、`version`）的
  SHA-256 指纹加时间终点为键：相同请求逐位复现；**模型版本不同或任一速率参数
  不同则键不同，结果不可能被错误复用**。缓存数组只读并用锁保护。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/validate` | 仅校验，返回 `{valid, fingerprint}` |
| POST | `/simulate` | 时间历程仿真，返回轨迹、总质量偏差、最小浓度 |
| POST | `/scan` | 单参数批量扫描，每项含独立指纹与轨迹 |

## 验证方法（测试）

```bash
uv run pytest --log-cli-level=INFO
```

- `tests/test_validation.py`：10 类非法网络（缺失物种/酶、非法计量、不可行速率、
  未闭合网络等）各映射到独立 code，并逐例核对 HTTP 与离线校验结论一致。
- `tests/test_simulation.py`：闭合网络守恒、可逆反应平衡点、输出步长细化不变性、
  分段与一次求值一致性、非负性；通过注入轨迹确定性地触发
  `NEGATIVE_CONCENTRATION` 与 `CONSERVATION_VIOLATION`。
- `tests/test_scanning.py`：扰动不改原网络、同请求逐位复现、版本指纹隔离、
  8 轮 × 5 参数值的并发扫描互不污染且各有独立指纹。
- `tests/test_api.py`：32 个并发 HTTP 请求轨迹逐元素一致、扫描指纹互不复用、
  错误信封机器可读。

每个测试用例均通过日志打印关键输入（网络/参数值/注入轨迹）与判定依据
（偏差数值、指纹、比较结果），失败信息可直接定位到原因 code 和具体反应/物种。
