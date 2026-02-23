#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分阶段验证统一运行脚本

运行所有三个阶段的验证，并生成汇总报告。

使用方法：
    python verify/run_all_verification.py

或分别运行：
    python verify/verify_phase1.py           # 引力场学习
    python verify/verify_phase2_optimized.py # 轨迹优化（多种方法）
    python verify/verify_phase3.py           # 控制与仿真
"""

import os
import sys
import subprocess
import time


def run_phase(phase_num, script_name):
    """运行单个阶段验证"""
    print("\n" + "=" * 70)
    print(f"🚀 开始运行第 {phase_num} 阶段验证")
    print("=" * 70)

    script_path = os.path.join("verify", script_name)

    if not os.path.exists(script_path):
        print(f"❌ 错误：找不到脚本 {script_path}")
        return False

    start_time = time.time()

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=False,
            text=True,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ 第 {phase_num} 阶段验证成功 (耗时: {elapsed:.1f}s)")
            return True
        else:
            print(f"\n❌ 第 {phase_num} 阶段验证失败 (耗时: {elapsed:.1f}s)")
            return False

    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        return False


def main():
    """主函数：运行所有验证"""
    print("=" * 70)
    print("🚀 小行星附着轨迹设计项目 - 分阶段验证")
    print("=" * 70)
    print("\n此脚本将依次运行三个阶段的验证：")
    print("  1️⃣  第一阶段：引力场学习")
    print("  2️⃣  第二阶段：轨迹优化")
    print("  3️⃣  第三阶段：控制与仿真")
    print("\n各阶段之间的数据会自动衔接")
    print("=" * 70)

    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"\n📂 工作目录: {os.getcwd()}")

    total_start = time.time()
    results = {}

    # 运行第一阶段
    results[1] = run_phase(1, "verify_phase1.py")

    if not results[1]:
        print("\n⚠️ 第一阶段失败，是否继续运行第二阶段？(y/n)")
        response = input().strip().lower()
        if response != "y":
            print("已取消后续验证")
            return

    # 运行第二阶段
    results[2] = run_phase(2, "verify_phase2_optimized.py")

    if not results[2]:
        print("\n⚠️ 第二阶段失败，是否继续运行第三阶段？(y/n)")
        response = input().strip().lower()
        if response != "y":
            print("已取消后续验证")
            return

    # 运行第三阶段
    results[3] = run_phase(3, "verify_phase3.py")

    # 汇总报告
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("📊 验证完成汇总")
    print("=" * 70)

    for i in range(1, 4):
        status = "✅ 通过" if results.get(i, False) else "❌ 失败"
        print(f"  第 {i} 阶段: {status}")

    print(f"\n⏱️  总耗时: {total_elapsed:.1f} 秒")

    # 检查是否有报告文件
    report_file = "results/phase3/final_report.txt"
    if os.path.exists(report_file):
        print(f"\n📄 详细报告: {report_file}")
        print("\n报告内容预览:")
        print("-" * 70)
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            print(f"无法读取报告: {str(e)}")
        print("-" * 70)

    print("\n" + "=" * 70)
    if all(results.values()):
        print("🎉 所有阶段验证通过！")
    else:
        print("⚠️ 部分阶段验证未通过，请检查输出日志")
    print("=" * 70)


if __name__ == "__main__":
    main()
