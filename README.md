# 10 分钟事件合约 ARIMA 预测提醒工具

基于 ARIMA 模型的 Binance 10 分钟事件合约 **预测提醒** 工具。本工具根据公开市场数据生成「涨 / 跌 / 观望」信号，并在置信度达到阈值时推送到 Telegram。

**重要：本工具只提供预测提醒，不构成投资建议，也不自动下单。** 完整说明见 [docs/RISK_DISCLAIMER.md](docs/RISK_DISCLAIMER.md)。

详细实现计划见 [docs/基于ARIMA模型的10分钟事件合约预测工具的实现plan.md](docs/基于ARIMA模型的10分钟事件合约预测工具的实现plan.md)。

---

## 环境准备

### 系统要求

- Windows 10/11（文档命令以 PowerShell 为例）
- [Anaconda](https://www.anaconda.com/) 或 Miniconda
- Python 3.10+（推荐在独立 Conda 环境中运行）
- 可访问 Binance 公开 API 的网络
- 可访问 `api.telegram.org` 的网络（仅在使用 Telegram 推送时需要）

### 创建并激活 `arima-env`

若尚未创建环境：

```powershell
conda create -n arima-env python=3.11 -y
conda activate arima-env
cd C:\dev\program\event
pip install -r requirements.txt
```

**每次**打开新终端后，先进入项目目录并激活环境（后续所有命令均默认已执行此步骤）：

```powershell
cd C:\dev\program\event
conda activate arima-env
```

若 PowerShell 无法识别 `conda activate`：

```powershell
conda init powershell
```

关闭并重新打开终端后再执行 `conda activate arima-env`。

### 安装依赖

```powershell
conda activate arima-env
pip install -r requirements.txt
```

### 验证环境

```powershell
conda activate arima-env
python -c "import pandas, statsmodels; print('OK')"
python -c "from src.utils.config import load_settings; print(load_settings().symbol)"
```

---

## `.env` 配置

### 创建配置文件

```powershell
conda activate arima-env
copy .env.example .env
```

使用文本编辑器打开 `.env`，按需填写。**不要将 `.env` 提交到 git。**

### 配置项说明

| 变量 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `SYMBOL` | 是 | 交易对 | `BTCUSDT` |
| `INTERVAL` | 是 | K 线周期 | `1m` |
| `PREDICTION_MINUTES` | 是 | 预测窗口（分钟） | `10` |
| `ARIMA_ORDER` | 是 | ARIMA 阶数 p,d,q | `1,0,1` |
| `ARIMA_SERIES_TYPE` | 否 | 建模序列：`log_return` 或 `price_diff` | `log_return` |
| `USE_AUTO_ARIMA` | 否 | 是否自动选参 | `false` |
| `TRAIN_WINDOW` | 是 | 训练窗口（1m K 线根数） | `1440` |
| `REFIT_INTERVAL_MINUTES` | 否 | 模型重拟合间隔（分钟） | `5` |
| `DIRECTION_THRESHOLD` | 否 | 涨跌方向幅度阈值 | `0.0` |
| `CONFIDENCE_THRESHOLD` | 是 | 推送最低置信度（0–1） | `0.70` |
| `SIGNAL_COOLDOWN_MINUTES` | 否 | 同方向信号冷却时间 | `10` |
| `MAX_SPREAD_BPS` | 否 | 盘口价差过滤（基点） | `50` |
| `BINANCE_MARKET` | 否 | 数据源：`spot` 或 `futures` | `spot` |
| `BINANCE_API_KEY` | 否 | 公开行情可不填 | |
| `BINANCE_API_SECRET` | 否 | 与 API Key 成对使用 | |
| `TELEGRAM_BOT_TOKEN` | 实时推送时 | Bot Token | |
| `TELEGRAM_CHAT_ID` | 实时推送时 | 接收方 Chat ID | |
| `DRY_RUN` | 否 | `true` 时不发真实 Telegram | `true` |
| `LOG_LEVEL` | 否 | 日志级别 | `INFO` |
| `LIVE_POLL_INTERVAL_SECONDS` | 否 | 实时轮询间隔（秒） | `10` |
| `DATA_DIR` / `LOGS_DIR` | 否 | 数据与日志目录 | `data` / `logs` |

### 最小可运行配置

仅下载历史数据或回测时，可不填 Telegram：

```env
SYMBOL=BTCUSDT
INTERVAL=1m
PREDICTION_MINUTES=10
ARIMA_ORDER=1,0,1
TRAIN_WINDOW=1440
CONFIDENCE_THRESHOLD=0.70
DRY_RUN=true
```

开启真实 Telegram 推送前，需补充 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`，并将 `DRY_RUN=false`（或使用 CLI `--no-dry-run`）。

---

## 历史数据下载

从 Binance 公开 API 分页下载 1 分钟 K 线，自动去重并检查时间连续性，保存到 `data/raw/`。

```powershell
conda activate arima-env
python -m src.data.download_klines --symbol BTCUSDT --interval 1m --start 2026-01-01 --end 2026-02-01
```

下载约 30 天数据：

```powershell
conda activate arima-env
python -m src.data.download_klines --symbol BTCUSDT --interval 1m --start 2026-01-01 --end 2026-01-31
```

常用参数：

- `--market spot|futures`：数据源（默认 `.env` 中 `BINANCE_MARKET`）
- `--output-dir data/raw`：输出目录
- `--min-interval 0.2`：请求间隔（秒），用于限频
- `-v`：调试日志

输出示例：`data/raw/BTCUSDT_1m.csv`

下载后检查：

- 时间戳是否基本连续（允许交易所维护间隙）
- 是否存在重复 `timestamp`
- `close`、`volume` 是否为空

---

## 回测

在开启实时 Telegram 推送前，**必须先**完成无未来函数的滚动回测。

```powershell
conda activate arima-env
python -m src.backtest.run_backtest --symbol BTCUSDT --data data/raw/BTCUSDT_1m.csv --prediction-minutes 10
```

可选：若已采集盘口数据，可附加 `--orderbook data/raw/BTCUSDT_orderbook.csv`。

结果写入 `data/backtest/`，终端会打印摘要。关注指标：

- 信号数与信号频率（是否过少/过多）
- 涨/跌信号胜率、总胜率、平衡准确率
- 最大连错次数
- 简化收益模拟（仅作参考，非实盘保证）

若胜率不稳定或接近随机，应提高 `CONFIDENCE_THRESHOLD` 或延长训练窗口，**不要**急于开启实时提醒。

---

## 实时运行

实时入口整合数据采集、ARIMA 预测、信号引擎与 Telegram 推送，支持日志、重试、Ctrl+C 优雅退出和 dry-run。

### 第一步：dry-run 验证

```powershell
conda activate arima-env
python -m src.app --mode live --dry-run
```

单轮测试（不进入持续循环）：

```powershell
conda activate arima-env
python -m src.app --mode live --dry-run --once
```

### 第二步：开启真实推送

确认日志与信号逻辑正常后：

```powershell
conda activate arima-env
python -m src.app --mode live --no-dry-run
```

或在 `.env` 设置 `DRY_RUN=false` 后：

```powershell
conda activate arima-env
python -m src.app --mode live
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--dry-run` | 强制只写日志，不调用 Telegram |
| `--no-dry-run` | 允许推送（覆盖 `.env` 中 `DRY_RUN=true`） |
| `--once` | 只执行一轮 |
| `--poll-interval 10` | 循环间隔（秒） |
| `--symbol` / `--interval` / `--market` | 覆盖 `.env` |
| `--no-health-check` | 跳过启动健康检查消息 |
| `-v` | 调试日志 |

仅采集行情（不跑 ARIMA 主循环）时，可使用：

```powershell
conda activate arima-env
python -m src.data.collect_live --once
python -m src.data.collect_live
```

---

## Telegram 测试

### 获取 Bot Token 与 Chat ID

1. 在 Telegram 中搜索 [@BotFather](https://t.me/BotFather)，创建 Bot 并复制 Token。
2. 向你的 Bot 发送 `/start`。
3. 使用 [@userinfobot](https://t.me/userinfobot) 或调用 `getUpdates` API 获取 `chat_id`。
4. 将二者写入 `.env` 的 `TELEGRAM_BOT_TOKEN` 与 `TELEGRAM_CHAT_ID`。

### 发送测试消息

不调用真实 API（仅写日志）：

```powershell
conda activate arima-env
python -m src.notify.telegram --test --dry-run
```

发送真实健康检查消息：

```powershell
conda activate arima-env
python -m src.notify.telegram --test
```

成功时，Telegram 应收到包含标的、周期、置信度阈值和「不自动下单」提示的启动消息。

---

## 日志查看

实时运行与采集模块会将日志写入 `logs/`：

| 文件 | 内容 |
|------|------|
| `logs/app.log` | 主程序、循环状态 |
| `logs/data.log` | K 线/盘口采集、API 重试 |
| `logs/model.log` | ARIMA 拟合与预测 |
| `logs/signal.log` | 信号过滤、冷却、置信度 |
| `logs/telegram.log` | 推送成功/失败、dry-run |

PowerShell 查看最近日志：

```powershell
Get-Content logs\app.log -Tail 50
Get-Content logs\model.log -Tail 30
```

排查顺序建议：数据采集 → 模型拟合 → 信号过滤 → Telegram 推送。

---

## 测试

项目使用 `pytest` 覆盖配置、下载、特征、ARIMA、信号、回测、Telegram、实时循环与应用入口。

### 运行全部测试

```powershell
conda activate arima-env
pytest
```

### 按模块运行

```powershell
conda activate arima-env
pytest tests/test_config.py -v
pytest tests/test_download_klines.py -v
pytest tests/test_features.py -v
pytest tests/test_arima_predictor.py -v
pytest tests/test_signal_engine.py -v
pytest tests/test_backtest.py -v
pytest tests/test_telegram.py -v
pytest tests/test_live_runner.py -v
pytest tests/test_app.py -v
```

### 测试范围说明

- **配置**：`.env` 解析、校验、敏感项缺失提示
- **数据**：K 线分页/去重、实时采集、盘口字段
- **特征**：无未来泄露、标签边界
- **模型**：固定阶数预测、失败降级
- **信号**：置信度、冷却、价差过滤
- **回测**：滚动窗口、指标汇总
- **通知**：消息格式含「不自动下单」「人工确认」
- **应用**：CLI dry-run 解析、单轮 live 冒烟

Telegram 相关测试默认 **mock HTTP**，不会向真实 Bot 发消息。

---

## 常见问题

### `conda activate` 无效

执行 `conda init powershell` 后重启终端；或从开始菜单打开 **Anaconda Prompt**，再 `conda activate arima-env`。

### `pip install` 或 `statsmodels` 安装失败

确认已激活 `arima-env`，并使用该环境下的 `pip`：`where python`、`where pip` 应指向 `arima-env`。

### 历史下载很慢或报限频

增大 `--min-interval`（如 `0.5`）；缩小单次下载日期范围，分多次下载后合并。

### 回测报错「缺少列」或「数据不足」

确认 CSV 包含完整 K 线字段；`TRAIN_WINDOW + PREDICTION_MINUTES` 应小于 K 线总行数；建议至少 30 天 `1m` 数据。

### ARIMA 频繁拟合失败

查看 `logs/model.log`；可增大 `TRAIN_WINDOW`、改用 `log_return`、关闭 `USE_AUTO_ARIMA`，或放宽 `REFIT_INTERVAL_MINUTES`。

### Telegram 测试收不到消息

- Token、Chat ID 是否正确（无多余空格）
- 是否已向 Bot 发送过 `/start`
- 本机网络能否访问 `api.telegram.org`
- 先用 `--dry-run` 确认程序逻辑，再测真实推送

### 实时运行无信号推送

- 多数时间为「观望」属正常；检查 `CONFIDENCE_THRESHOLD` 是否过高
- 确认 `DRY_RUN` 未误设为 `true`，且未同时传入 `--dry-run`
- 查看 `logs/signal.log` 中的过滤原因（冷却、价差、成交量等）

### 信号与回测表现差异大

实时行情分布可能与历史样本不同；建议每周用最新数据重新回测，剧烈行情时暂停推送。

---

## 项目结构

```text
event/
  data/
    raw/          # 原始 K 线、盘口
    processed/    # 特征数据
    backtest/     # 回测结果
  docs/
    基于ARIMA模型的10分钟事件合约预测工具的实现plan.md
    RISK_DISCLAIMER.md
  logs/
  src/
    data/         # 下载与实时采集
    features/     # 特征与标签
    models/       # ARIMA
    signals/      # 信号引擎
    notify/       # Telegram
    backtest/     # 滚动回测
    app.py        # 实时入口
    live_runner.py
    utils/        # 配置
  tests/
  .env.example
  requirements.txt
  README.md
```

---

## 风险提示

**本工具只提供预测提醒，不构成投资建议，也不自动下单。**

- 所有信号均需用户 **人工确认** 后再决定是否参与事件合约。
- ARIMA 对短周期加密市场的非线性波动、消息面冲击捕捉能力有限。
- 历史回测胜率 **不保证** 未来表现；上线实时推送前须完成无未来函数回测。
- 用户须自行承担交易风险，并遵守当地法规与 Binance 服务条款。

完整条款见 **[docs/RISK_DISCLAIMER.md](docs/RISK_DISCLAIMER.md)**。
