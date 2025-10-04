import psutil
import time
import datetime

# 日志文件
LOG_FILE = "cpu_log.txt"

def log_high_cpu_processes(threshold=10, top_n=10):
    """
    threshold: 占用 CPU 百分比的最低阈值
    top_n: 记录前 N 个进程
    """
    # 获取所有进程
    processes = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            cpu = p.info['cpu_percent']
            if cpu is None:
                continue
            processes.append((cpu, p.info['pid'], p.info['name']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 按 CPU 使用率排序
    processes.sort(reverse=True, key=lambda x: x[0])

    # 过滤超过阈值的
    high_cpu = [p for p in processes if p[0] >= threshold][:top_n]

    if high_cpu:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.datetime.now()} ===\n")
            for cpu, pid, name in high_cpu:
                f.write(f"PID {pid:5d} | CPU {cpu:5.1f}% | {name}\n")


if __name__ == "__main__":
    print("开始记录 CPU 占用情况，按 Ctrl+C 停止")
    # 先预热一下（psutil 第一次调用 cpu_percent 会是0）
    psutil.cpu_percent(interval=1)

    while True:
        log_high_cpu_processes(threshold=10, top_n=10)
        time.sleep(5)  # 每 5 秒记录一次
