# 10 分钟事件合约 ARIMA 预测提醒工具

基于 ARIMA 模型的 Binance 10 分钟事件合约预测提醒工具。本工具**不自动下单**，仅根据公开市场数据生成「涨 / 跌 / 观望」信号，并在置信度达到阈值时推送到 Telegram。

详细实现计划见 [docs/基于ARIMA模型的10分钟事件合约预测工具的实现plan.md](docs/基于ARIMA模型的10分钟事件合约预测工具的实现plan.md)。

## 环境要求

- Python 3.10+
- Anaconda 环境 `arima-env`

## 快速开始

### 1. 激活 Conda 环境

每次打开终端后，先进入项目目录并激活环境：

```powershell
cd C:\dev\program\event
conda activate arima-env
```

若 PowerShell 无法识别 `conda activate`，先执行 Anaconda 初始化并重新打开终端：

```powershell
conda init powershell
```

### 2. 安装依赖

```powershell
conda activate arima-env
pip install -r requirements.txt
```

### 3. 准备配置

复制示例配置并填写敏感信息：

```powershell
conda activate arima-env
copy .env.example .env
```

在 `.env` 中至少配置：

- `SYMBOL`：交易标的，默认 `BTCUSDT`
- `INTERVAL`：K 线周期，默认 `1m`
- `PREDICTION_MINUTES`：预测窗口，默认 `10`
- `ARIMA_ORDER`：ARIMA 阶数，例如 `1,0,1`
- `TRAIN_WINDOW`：训练窗口长度（1 分钟 K 线根数）
- `CONFIDENCE_THRESHOLD`：信号置信度阈值
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`：Telegram 推送（实时模式需要）

`.env` 含敏感信息，**不要提交到 git**。

### 4. 验证配置加载

```powershell
conda activate arima-env
python -c "from src.utils.config import load_settings; print(load_settings())"
```

## 项目结构

```text
event/
  data/
    raw/          # 原始 K 线、盘口数据
    processed/    # 清洗后的训练数据
    backtest/     # 回测结果
  docs/           # 设计文档
  logs/           # 运行日志
  src/
    data/         # 数据采集
    features/     # 特征工程
    models/       # ARIMA 建模
    signals/      # 信号引擎
    notify/       # Telegram 推送
    backtest/     # 回测
    live_runner.py # 实时运行循环
    app.py        # 实时运行入口
    utils/        # 工具与配置
  tests/          # 单元测试
  .env.example    # 配置示例
  requirements.txt
  README.md
```

## 下载历史 K 线

从 Binance 公开 API 分页下载 1 分钟 K 线，自动去重并检查时间连续性，保存到 `data/raw/`。所有命令均需在 `arima-env` 中执行。

```powershell
conda activate arima-env
python -m src.data.download_klines --symbol BTCUSDT --interval 1m --start 2026-01-01 --end 2026-02-01
```

下载约 30 天数据示例：

```powershell
conda activate arima-env
python -m src.data.download_klines --symbol BTCUSDT --interval 1m --start 2026-01-01 --end 2026-01-31
```

可选参数：

- `--market spot|futures`：数据源（默认读取 `.env` 中 `BINANCE_MARKET`）
- `--output-dir`：输出目录（默认 `data/raw`）
- `--min-interval 0.2`：请求间隔秒数，用于限频
- `-v`：输出调试日志

输出文件：`data/raw/BTCUSDT_1m.csv`

运行测试：

```powershell
conda activate arima-env
pytest tests/test_download_klines.py -v
```

## 实时行情采集

使用 REST 轮询获取最新 1 分钟 K 线与买卖盘最优价量，追加保存到 `data/raw/`。数据源接口支持后续替换为 WebSocket。

单次采集（验证连通性）：

```powershell
conda activate arima-env
python -m src.data.collect_live --once
```

持续采集（默认每 10 秒轮询，Ctrl+C 优雅退出）：

```powershell
conda activate arima-env
python -m src.data.collect_live
```

输出文件：

- `data/raw/BTCUSDT_1m.csv`：K 线（与历史下载共用，按 timestamp 去重合并）
- `data/raw/BTCUSDT_orderbook.csv`：盘口快照（含 spread、mid_price、book_imbalance）

可选参数：

- `--poll-interval 10`：轮询间隔秒数
- `--kline-limit 2`：每次拉取最近 K 线根数
- `--market spot|futures`：数据源
- `--once`：只执行一轮后退出
- `-v`：调试日志

日志写入 `logs/data.log`（可用 `--no-log-file` 关闭文件日志）。

运行测试：

```powershell
conda activate arima-env
pytest tests/test_collect_live.py -v
```

## 运行回测

```powershell
conda activate arima-env
python -m src.backtest.run_backtest --symbol BTCUSDT --data data/raw/BTCUSDT_1m.csv --prediction-minutes 10
```

回测结果写入 `data/backtest/`。建议先在不少于 30 天的 1 分钟数据上验证胜率后再开启实时推送。

运行测试：

```powershell
conda activate arima-env
pytest tests/test_backtest.py -v
```

## 测试 Telegram

```powershell
conda activate arima-env
python -m src.notify.telegram --test
```

Dry-run 测试（只写日志，不调用 Telegram API）：

```powershell
conda activate arima-env
python -m src.notify.telegram --test --dry-run
```

## 实时运行

实时入口整合数据采集、ARIMA 预测、信号引擎与 Telegram 推送。支持日志、异常重试、优雅退出（Ctrl+C）和 dry-run 模式。

**建议先用 dry-run 验证数据、模型和日志：**

```powershell
conda activate arima-env
python -m src.app --mode live --dry-run
```

单轮测试（不进入持续循环）：

```powershell
conda activate arima-env
python -m src.app --mode live --dry-run --once
```

确认无误后，开启真实 Telegram 推送：

```powershell
conda activate arima-env
python -m src.app --mode live --no-dry-run
```

或在 `.env` 中设置 `DRY_RUN=false` 后直接运行：

```powershell
conda activate arima-env
python -m src.app --mode live
```

可选参数：

- `--poll-interval 10`：每轮循环间隔秒数（默认读取 `LIVE_POLL_INTERVAL_SECONDS`）
- `--symbol BTCUSDT` / `--interval 1m` / `--market spot|futures`：覆盖 `.env` 配置
- `--once`：只执行一轮后退出
- `--no-health-check`：跳过启动时的 Telegram 健康检查
- `-v`：调试日志

运行日志：

- `logs/app.log`：主程序
- `logs/data.log`：数据采集
- `logs/model.log`：ARIMA 预测
- `logs/signal.log`：信号过滤
- `logs/telegram.log`：Telegram 推送

运行测试：

```powershell
conda activate arima-env
pytest tests/test_live_runner.py -v
```

## 风险提示

- 本工具仅提供预测提醒，**不构成投资建议**，也**不自动下单**。
- ARIMA 对短周期加密市场的非线性波动捕捉能力有限。
- 上线实时推送前，必须先完成无未来函数的历史回测。
- 用户自行承担事件合约参与风险，并遵守当地监管与 Binance 服务条款。
