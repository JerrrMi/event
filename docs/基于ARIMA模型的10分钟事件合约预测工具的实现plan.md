# 基于 ARIMA 模型的 10 分钟事件合约预测工具实现计划

## 1. 项目目标

本工具用于辅助 Binance 10 分钟事件合约的人工决策。首版只做预测与 Telegram 信号提醒，不自动下单、不托管资金、不调用事件合约交易接口。

核心工作流：

```text
抓取 Binance 1 分钟 K 线
  -> 清洗并生成价格/收益率序列
  -> 使用 ARIMA 模型滚动预测未来 10 分钟方向
  -> 当预测方向为涨/跌且置信度超过阈值时生成开仓信号
  -> 将信号推送到 Telegram
```

首版目标是先做一个可验证、可回测、可长期运行的信号工具，而不是追求复杂模型。ARIMA 的优势是实现简单、推理快、可解释，适合作为 10 分钟预测工具的统计基线。

## 2. 范围与假设

### 2.1 首版包含

- 使用 Python 实现命令行工具和常驻信号服务。
- 从 Binance 公共 Kline API 拉取 1 分钟 K 线。
- 支持历史数据回填、本地存储和增量更新。
- 使用 `statsmodels` 的 ARIMA/SARIMAX 进行滚动窗口预测。
- 输出未来 10 根 1 分钟 K 线的累计方向预测。
- 根据信号置信度、最小预测幅度、冷却时间等规则决定是否提醒。
- 通过 Telegram Bot API 向指定 chat 推送信号。
- 提供基础回测，用于校准 ARIMA 参数、置信度阈值和信号频率。

### 2.2 首版不包含

- 自动下单或任何资金操作。
- 事件合约私有 API 集成。
- 盘口、逐笔成交、新闻情绪、链上指标等高阶特征。
- LSTM、XGBoost、Transformer、强化学习等非 ARIMA 模型。
- 多用户权限系统、Web 后台和复杂可视化界面。

### 2.3 默认实现假设

- 默认交易对：`BTCUSDT`，但必须支持通过配置切换。
- 默认周期：`1m` K 线。
- 默认预测期限：`10` 分钟。
- 默认训练窗口：最近 `1000-3000` 根 1 分钟 K 线，具体通过回测校准。
- 默认信号阈值：置信度从 `0.60-0.65` 起步，最小预测幅度从 `0.03%-0.08%` 起步，实际值必须由回测结果决定。
- Telegram Token、Chat ID、交易对、阈值等敏感或可变参数都放在 `.env` 或配置文件中，不写死在代码里。

## 3. 推荐技术栈

- Python 3.11 或更高版本。
- `pandas`：K 线数据处理和时间序列整理。
- `numpy`：数值计算。
- `statsmodels`：ARIMA/SARIMAX 建模。
- `requests` 或 `httpx`：调用 Binance 和 Telegram HTTP API。
- `python-dotenv` 或 `pydantic-settings`：加载环境变量。
- `typer` 或 `argparse`：命令行入口。
- `pytest`：测试。
- `ruff`：代码格式和静态检查。
- `pyarrow`：可选，用于 Parquet 存储。

建议本地数据优先使用 `data/klines/{symbol}_1m.parquet`。如果项目希望更轻量，也可以先用 CSV，但 Parquet 在增量读写、体积和类型保持上更适合后续扩展。

## 4. 目录结构建议

```text
.
├── .env.example
├── README.md
├── requirements.txt 或 pyproject.toml
├── data/
│   ├── klines/
│   ├── predictions/
│   └── backtests/
├── src/
│   └── event_predictor/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── binance_client.py
│       ├── storage.py
│       ├── features.py
│       ├── arima_model.py
│       ├── signal.py
│       ├── telegram.py
│       ├── backtest.py
│       └── scheduler.py
└── tests/
    ├── test_features.py
    ├── test_signal.py
    └── test_backtest.py
```

## 5. 数据流程设计

### 5.1 Binance 1 分钟 K 线抓取

数据源使用 Binance 公共 REST Kline 接口：

- Spot Kline：`GET /api/v3/klines`
- Futures Kline：`GET /fapi/v1/klines`

首版建议优先使用与事件合约标的最接近、可稳定访问的数据源。若事件合约参考的是某个指数或合约价格，应在 README 中明确说明当前使用的替代价格源。

每根 K 线至少保留：

- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `close_time`
- `quote_volume`
- `trade_count`

实时服务必须只使用已经闭合的 1 分钟 K 线，避免把未完成 K 线当成确定价格，从而造成未来函数或信号漂移。

### 5.2 本地存储

本地存储要求：

- 按 `symbol + interval` 分文件保存。
- 使用 `open_time` 去重。
- 每次增量拉取后按时间排序。
- 对缺失分钟做检测，发现缺口时记录日志并尝试补齐。
- 所有时间统一使用 UTC。

### 5.3 数据清洗

清洗规则：

- 将价格字段转成浮点数或 `Decimal` 后再计算。
- 丢弃重复 K 线。
- 检查时间间隔是否严格为 1 分钟。
- 对缺失 K 线不做随意插值；优先重新拉取，无法补齐时在建模窗口中跳过该区间。
- 计算 log price 和 log return：

```text
log_price_t = log(close_t)
log_return_t = log(close_t / close_{t-1})
```

ARIMA 可直接建模价格差分，也可建模 log return。首版建议以 log return 为主，因为短期价格通常非平稳，而收益率序列更接近平稳。

## 6. ARIMA 建模方案

### 6.1 预测目标

在每个信号评估时刻 `t`：

- 输入：截至 `t` 已闭合的 1 分钟 K 线。
- 预测：未来 10 分钟累计收益 `R_{t,t+10}`。
- 输出：
  - `direction = UP`：预测累计收益大于 0。
  - `direction = DOWN`：预测累计收益小于 0。
  - `confidence`：方向判断的估计置信度。
  - `expected_return`：预测累计收益。
  - `target_price`：预测到期参考价格。

### 6.2 模型输入

首版推荐输入序列：

```text
y_t = log_return_t
```

使用最近 `N` 根 1 分钟收益率作为滚动训练窗口。`N` 可从 `1000`、`2000`、`3000` 三档开始回测比较。

### 6.3 ARIMA 参数

先从小范围网格搜索开始：

- `p`: `0-5`
- `d`: 对 log return 通常为 `0`；如果建模 log price，则通常为 `1`
- `q`: `0-5`

参数选择方式：

- 初始可按 AIC/BIC 自动选取。
- 实盘服务中不建议每分钟全量网格搜索，避免耗时和不稳定。
- 推荐每日或每数小时重新选择一次参数，实时预测使用最近选出的 `(p, d, q)`。
- 如果模型拟合失败，记录错误并跳过本轮信号，不发送低质量提醒。

### 6.4 10 分钟预测

对未来 10 步收益率做预测：

```text
predicted_returns = forecast(steps=10)
expected_return = sum(predicted_returns)
target_price = current_close * exp(expected_return)
```

方向判断：

```text
if expected_return > 0: UP
if expected_return < 0: DOWN
```

如果 `expected_return` 的绝对值过小，即使置信度达标也应过滤，因为事件合约方向预测需要覆盖噪声、延迟和价差影响。

### 6.5 置信度估算

ARIMA 本身输出的是预测均值和置信区间，不是直接的二分类概率。首版可用以下方式估算方向置信度：

1. 获取未来 10 步累计预测收益的均值 `mu` 和标准差 `sigma`。
2. 假设预测误差近似正态分布。
3. 对上涨方向：

```text
confidence_up = P(R > 0) = 1 - CDF(0, mu, sigma)
```

4. 对下跌方向：

```text
confidence_down = P(R < 0) = CDF(0, mu, sigma)
```

最终：

```text
confidence = max(confidence_up, confidence_down)
```

如果无法可靠取得预测方差，可以用滚动回测残差估计 `sigma`。此时必须在文档和日志中标注该置信度是经验估计，不是保证胜率。

## 7. 信号生成规则

每分钟评估一次，但不一定每分钟发信号。建议规则：

- 只在新闭合 1 分钟 K 线后运行预测。
- `confidence >= MIN_CONFIDENCE` 才允许发信号。
- `abs(expected_return) >= MIN_EXPECTED_RETURN` 才允许发信号。
- 同一交易对发出信号后进入冷却期，例如 `10-20` 分钟。
- 如果最近连续失败次数过多，可进入暂停期。
- 如果数据缺口、模型拟合失败、Telegram 发送失败，不生成开仓信号，只记录日志。

信号对象建议包含：

```text
symbol
interval
horizon_minutes
direction
confidence
expected_return
current_price
target_price
model_order
train_window
generated_at
expires_at
reason
```

## 8. Telegram 推送设计

Telegram 消息应简洁但包含人工决策所需信息。

推荐格式：

```text
Binance 10分钟事件合约信号

交易对: BTCUSDT
方向: UP / 看涨
置信度: 64.2%
当前参考价: 68250.10
预测10分钟后: 68308.45
预测收益: +0.085%
模型: ARIMA(2,0,2)
训练窗口: 2000根1m K线
信号时间: 2026-05-22 02:30:00 UTC
有效期: 约10分钟

提示: 该信号仅供人工参考，不自动下单。事件合约可能损失全部投入本金。
```

Telegram 配置：

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_TIMEOUT_SECONDS`

安全要求：

- `.env` 不提交到 git。
- 提供 `.env.example` 作为模板。
- 日志中不得打印完整 Bot Token。

## 9. 回测与评估

回测必须在实盘提醒前完成。回测的目的不是证明策略一定盈利，而是校准阈值、发现过拟合和估计信号频率。

### 9.1 标签定义

对每个历史时刻 `t`：

```text
future_return_10m = log(close_{t+10} / close_t)
label = UP if future_return_10m > 0 else DOWN
```

如果 future return 非常接近 0，可以单独统计为 `FLAT` 或从训练评估中剔除，但事件合约最终仍只有涨/跌结果，因此该规则要在文档中固定。

### 9.2 回测方式

- 使用滚动窗口训练，严格只使用 `t` 之前的数据。
- 每次预测未来 10 分钟。
- 应用与实盘完全一致的信号门控规则。
- 统计所有信号，而不是统计每一分钟的裸预测。

### 9.3 关键指标

- 预测准确率。
- 上涨/下跌分别的准确率。
- 信号次数和日均信号次数。
- 平均置信度。
- 不同置信度分桶下的胜率。
- 最大连续失败次数。
- 简化收益曲线。
- 与随机 50% 基准的差异。

建议验收门槛：

- 回测流程无未来数据泄露。
- 高置信度分桶胜率应明显高于 50%，否则不建议上线提醒。
- 信号频率不能过高；如果一天几十上百条，说明阈值过低或冷却期不足。
- 在不同月份、不同市场波动环境下表现不能严重失衡。

## 10. 实时运行设计

实时服务循环：

```text
1. 等待下一根 1 分钟 K 线闭合
2. 拉取最近 K 线并更新本地数据
3. 检查数据完整性
4. 使用滚动窗口拟合或加载 ARIMA 参数
5. 预测未来 10 分钟累计收益
6. 计算方向、置信度和预测幅度
7. 应用信号门控规则
8. 若通过门控，推送 Telegram
9. 写入预测和信号日志
10. 进入下一轮
```

运行要求：

- 每轮任务应在数秒内完成。
- Binance 请求失败要重试，但不能无限阻塞。
- 模型拟合失败要跳过本轮。
- Telegram 失败要记录错误，并可在下一轮继续运行。
- 服务重启后应能从本地数据继续增量更新。

## 11. 配置项设计

建议 `.env.example` 包含：

```text
BINANCE_MARKET=spot
BINANCE_SYMBOL=BTCUSDT
BINANCE_INTERVAL=1m
PREDICTION_HORIZON_MINUTES=10
TRAIN_WINDOW=2000
ARIMA_ORDER=2,0,2
AUTO_SELECT_ARIMA_ORDER=false
MIN_CONFIDENCE=0.63
MIN_EXPECTED_RETURN=0.0005
SIGNAL_COOLDOWN_MINUTES=10
DATA_DIR=data
TELEGRAM_BOT_TOKEN=replace_me
TELEGRAM_CHAT_ID=replace_me
LOG_LEVEL=INFO
```

配置加载要求：

- 启动时校验必要配置。
- 对数值配置做范围检查。
- Token 缺失时允许跑回测，但不允许启动 Telegram 实时提醒。

## 12. Cursor 分步实现 Prompts

下面 prompts 可按顺序复制给 Cursor 执行。每一步完成后先运行相关测试或命令，再进入下一步。

### Prompt 1：初始化 Python 项目骨架

```text
请在当前仓库中初始化一个 Python 项目，用于 Binance 10 分钟事件合约 ARIMA 预测信号工具。

要求：
1. 使用 src layout，包名为 event_predictor。
2. 添加 requirements.txt 或 pyproject.toml，依赖包含 pandas、numpy、statsmodels、requests 或 httpx、python-dotenv、typer、pytest、ruff。
3. 创建 .env.example，包含 Binance、ARIMA、信号阈值、Telegram 相关配置。
4. 创建 README.md，简要说明项目目标：只做 Telegram 信号提醒，不自动下单。
5. 不实现业务逻辑，只建立目录、配置模板和最小 CLI 入口。
6. 完成后运行基础导入检查或测试。
```

### Prompt 2：实现配置加载

```text
请实现 event_predictor.config 模块。

要求：
1. 从 .env 和环境变量读取配置。
2. 支持 BINANCE_MARKET、BINANCE_SYMBOL、BINANCE_INTERVAL、PREDICTION_HORIZON_MINUTES、TRAIN_WINDOW、ARIMA_ORDER、MIN_CONFIDENCE、MIN_EXPECTED_RETURN、SIGNAL_COOLDOWN_MINUTES、DATA_DIR、TELEGRAM_BOT_TOKEN、TELEGRAM_CHAT_ID。
3. 对数值范围做校验，例如置信度必须在 0.5 到 1.0 之间，预测周期必须为正整数。
4. ARIMA_ORDER 从 "p,d,q" 字符串解析成三元组。
5. 添加单元测试覆盖默认值、非法值和 ARIMA_ORDER 解析。
```

### Prompt 3：实现 Binance K 线客户端

```text
请实现 event_predictor.binance_client 模块，用 Binance 公共 REST API 抓取 1 分钟 K 线。

要求：
1. 支持 spot 和 futures 两种 market 配置。
2. 函数输入 symbol、interval、start_time、end_time、limit。
3. 输出 pandas DataFrame，字段至少包含 open_time、open、high、low、close、volume、close_time、quote_volume、trade_count。
4. 所有时间统一为 UTC datetime。
5. 只返回已经闭合的 K 线。
6. 对 HTTP 错误、限流、空响应做清晰异常处理。
7. 添加测试，可 mock HTTP 响应，不依赖真实网络。
```

### Prompt 4：实现本地存储与历史回填

```text
请实现 event_predictor.storage 和 CLI 的历史数据回填命令。

要求：
1. 将 K 线保存到 data/klines/{symbol}_{interval}.parquet。
2. 读取已有数据后按 open_time 去重并排序。
3. 支持 backfill 命令：给定 symbol、interval、开始时间、结束时间，分批拉取 Binance K 线并保存。
4. 检测 1 分钟 K 线缺口，并在日志中输出缺口数量和范围。
5. 添加单元测试覆盖去重、排序、缺口检测。
```

### Prompt 5：实现特征与 ARIMA 模型

```text
请实现 event_predictor.features 和 event_predictor.arima_model。

要求：
1. features 中基于 close 计算 log_price 和 log_return。
2. arima_model 使用 statsmodels ARIMA 或 SARIMAX。
3. 输入最近 TRAIN_WINDOW 根 K 线，基于 log_return 拟合模型。
4. 预测未来 10 步 log_return，输出 expected_return、target_price、prediction_std、model_order。
5. 使用预测均值和标准差估算 UP/DOWN 方向置信度。
6. 模型拟合失败时抛出可识别异常，不生成信号。
7. 添加单元测试覆盖方向判断、收益率计算、异常路径。
```

### Prompt 6：实现信号门控

```text
请实现 event_predictor.signal 模块。

要求：
1. 定义 Signal 数据结构，包含 symbol、direction、confidence、expected_return、current_price、target_price、model_order、generated_at、expires_at、reason。
2. 实现信号门控规则：MIN_CONFIDENCE、MIN_EXPECTED_RETURN、SIGNAL_COOLDOWN_MINUTES。
3. 当置信度不足、预测幅度不足、冷却期未结束时，不生成开仓信号，并返回明确原因。
4. 添加单元测试覆盖通过、置信度不足、幅度不足、冷却期过滤。
```

### Prompt 7：实现 Telegram 推送

```text
请实现 event_predictor.telegram 模块。

要求：
1. 使用 Telegram Bot API sendMessage 发送消息。
2. 消息包含交易对、方向、置信度、当前参考价、预测10分钟后价格、预测收益、模型参数、训练窗口、信号时间、有效期和风险提示。
3. Token 和 Chat ID 从配置读取。
4. 日志中不得打印完整 Bot Token。
5. 发送失败时抛出清晰异常，实时服务可捕获后继续运行。
6. 添加测试，mock HTTP 请求并验证消息内容。
```

### Prompt 8：实现回测命令

```text
请实现 event_predictor.backtest 和 CLI 的 backtest 命令。

要求：
1. 从本地 K 线文件读取历史数据。
2. 使用滚动窗口方式训练 ARIMA，严格只使用预测时刻之前的数据。
3. 每次预测未来 10 分钟方向，并应用与实盘一致的信号门控规则。
4. 输出准确率、信号次数、日均信号次数、UP/DOWN 分方向准确率、不同置信度分桶胜率、最大连续失败次数。
5. 将结果保存到 data/backtests/。
6. 添加测试，重点验证没有未来数据泄露。
```

### Prompt 9：实现实时信号服务

```text
请实现 event_predictor.scheduler 和 CLI 的 run 命令。

要求：
1. 服务每分钟在 K 线闭合后运行一次。
2. 增量拉取最新已闭合 K 线并更新本地存储。
3. 检查数据完整性。
4. 调用 ARIMA 模型生成预测。
5. 调用 signal 模块判断是否需要提醒。
6. 通过 Telegram 推送通过门控的信号。
7. 将每次预测和信号结果保存到 data/predictions/。
8. HTTP、模型、Telegram 任一环节失败时记录日志并进入下一轮，不让服务崩溃。
```

### Prompt 10：补齐文档、测试和运行检查

```text
请完善 README.md 和测试。

要求：
1. README 写清楚安装、配置 .env、历史回填、回测、启动实时服务、Telegram 验证方法。
2. README 明确说明本工具只提供人工参考信号，不自动下单，事件合约可能损失全部投入本金。
3. 添加或补齐关键单元测试。
4. 运行 pytest 和 ruff，修复发现的问题。
5. 确认 .env 不会被提交，并提供 .env.example。
```

## 13. 实现完成后的使用方法

以下命令名称为建议形式，实际以最终 CLI 实现为准。

### 13.1 准备环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果项目使用 `pyproject.toml`：

```powershell
pip install -e .
```

### 13.2 配置环境变量

复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```text
BINANCE_MARKET=spot
BINANCE_SYMBOL=BTCUSDT
BINANCE_INTERVAL=1m
PREDICTION_HORIZON_MINUTES=10
TRAIN_WINDOW=2000
ARIMA_ORDER=2,0,2
MIN_CONFIDENCE=0.63
MIN_EXPECTED_RETURN=0.0005
SIGNAL_COOLDOWN_MINUTES=10
TELEGRAM_BOT_TOKEN=你的telegram_bot_token
TELEGRAM_CHAT_ID=你的chat_id
```

Telegram Bot 获取方式：

1. 在 Telegram 中找 `@BotFather` 创建 bot。
2. 获取 Bot Token。
3. 给 bot 发送一条消息，或把 bot 加入目标群。
4. 使用 Telegram API 或项目提供的测试命令获取 `chat_id`。
5. 将 Token 和 Chat ID 写入 `.env`。

### 13.3 回填历史 K 线

建议至少回填 30 天以上 1 分钟 K 线；如果要做更稳定的回测，建议回填 3-12 个月。

```powershell
python -m event_predictor.cli backfill --symbol BTCUSDT --interval 1m --start 2026-01-01 --end 2026-05-22
```

检查输出中是否存在缺口。如果缺口较多，应先补齐数据再回测。

### 13.4 运行回测

```powershell
python -m event_predictor.cli backtest --symbol BTCUSDT --interval 1m --horizon 10 --train-window 2000
```

重点查看：

- 高置信度信号胜率是否明显高于 50%。
- 每天信号数量是否合理。
- 最大连续失败次数是否可接受。
- UP 和 DOWN 是否严重偏向某一侧。
- 不同月份表现是否稳定。

如果结果不理想，优先调整：

- `ARIMA_ORDER`
- `TRAIN_WINDOW`
- `MIN_CONFIDENCE`
- `MIN_EXPECTED_RETURN`
- `SIGNAL_COOLDOWN_MINUTES`

### 13.5 测试 Telegram 推送

```powershell
python -m event_predictor.cli test-telegram
```

预期结果：Telegram 收到一条测试消息。若失败，检查：

- Bot Token 是否正确。
- Chat ID 是否正确。
- bot 是否有群发言权限。
- 本地网络是否能访问 Telegram API。

### 13.6 启动实时信号服务

```powershell
python -m event_predictor.cli run
```

服务启动后应每分钟检查一次最新闭合 K 线。只有当 ARIMA 预测方向明确且满足置信度阈值时，Telegram 才会收到开仓参考信号。

建议先用小窗口观察至少数天，不要直接根据未验证信号投入大额资金。

### 13.7 常见问题排查

#### 没有收到任何信号

- `MIN_CONFIDENCE` 可能过高。
- `MIN_EXPECTED_RETURN` 可能过高。
- 冷却时间过长。
- 当前行情震荡，ARIMA 没有给出足够明确方向。
- Telegram 配置不正确。

#### 信号过多

- 提高 `MIN_CONFIDENCE`。
- 提高 `MIN_EXPECTED_RETURN`。
- 增加 `SIGNAL_COOLDOWN_MINUTES`。
- 检查是否对未闭合 K 线重复生成信号。

#### 回测胜率接近 50%

- 当前 ARIMA 模型可能没有有效预测力。
- 训练窗口或 ARIMA 参数不合适。
- 10 分钟周期噪声过大。
- 需要先降低实盘预期，继续作为研究基线，而不是贸然使用。

#### 模型经常拟合失败

- 检查 K 线是否缺失或重复。
- 缩短或延长训练窗口。
- 限制 ARIMA 参数阶数。
- 对异常价格或极端收益做清洗。

## 14. 验收标准

实现完成后，至少满足以下条件才算可进入观察运行：

- 可以成功回填指定交易对的 1 分钟 K 线。
- 本地数据按时间排序、无重复，并能报告缺口。
- ARIMA 预测只使用历史数据，没有未来数据泄露。
- 回测结果能输出信号胜率、信号数量、置信度分桶胜率和最大连续失败次数。
- 低置信度、低预测幅度、冷却期内的预测不会推送 Telegram。
- Telegram 测试消息和真实信号消息都能送达。
- `.env` 不提交，Token 不出现在日志中。
- 实时服务遇到网络错误、模型错误、Telegram 错误时不会直接崩溃。
- README 中明确包含安装、配置、回填、回测、启动和风险提示。

## 15. 风险提示

ARIMA 是线性统计模型，无法稳定捕捉加密市场中的突发新闻、盘口冲击和非线性行为。10 分钟事件合约噪声很高，即使短期回测有效，也可能在市场状态变化后迅速失效。

该工具输出的 Telegram 消息只能作为人工参考，不代表确定收益。事件合约可能损失全部投入本金。上线前必须先进行充分回测和小额观察，并设置每日最大亏损、最大交易次数和暂停条件。
