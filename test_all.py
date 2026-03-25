#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整测试 - 测试所有优化方法（使用点质量引力模型）"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SimpleAsteroid:
    """简化的点质量引力模型"""
    def __init__(self, mu=11.0, omega=None):
        self.mu = mu
        self.omega = omega if omega is not None else np.array([0.0, 0.0, 3.6e-4])
    
    def compute_gravity(self, position):
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.zeros(3)
        return -self.mu * position / (r**3)


class SimpleSpacecraft:
    """简化的航天器模型"""
    def __init__(self):
        self.T_max = 20.0
        self.I_sp = 300.0
        self.g0 = 9.80665
        self.m0 = 500.0


def test_method(name, create_optimizer, r0, v0, m0, rf, vf, t_span):
    """测试单个方法"""
    print(f"\n测试 {name}...")
    start = time.time()
    try:
        opt = create_optimizer()
        res = opt.optimize(r0, v0, m0, rf, vf, t_span)
        elapsed = time.time() - start
        
        if res and 'r' in res:
            pos_err = np.linalg.norm(res['r'][-1] - rf)
            vel_err = np.linalg.norm(res['v'][-1] - vf)
            fuel = m0 - res['m'][-1]
            success = res.get('success', False)
            
            status = "成功" if success else "部分"
            print(f"[{status}] {name}: 位置={pos_err:.4f}m, 速度={vel_err:.4f}m/s, 燃料={fuel:.2f}kg, 耗时={elapsed:.2f}s")
            return success, {'pos_err': pos_err, 'vel_err': vel_err, 'fuel': fuel}, elapsed
        else:
            print(f"[失败] {name}: 无有效结果, 耗时={elapsed:.2f}s")
            return False, None, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"[异常] {name}: {str(e)[:80]}, 耗时={elapsed:.2f}s")
        return False, None, elapsed


def main():
    asteroid = SimpleAsteroid(mu=11.0)
    spacecraft = SimpleSpacecraft()
    
    # 问题设置
    r0 = np.array([500.0, 0.0, 100.0])
    v0 = np.array([0.0, 0.2, 0.0])
    m0 = spacecraft.m0
    rf = np.array([400.0, 50.0, 50.0])
    vf = np.array([0.0, 0.1, 0.0])
    t_span = [0.0, 1800.0]
    
    print("=" * 60)
    print("小行星附着轨迹优化 - 完整测试（点质量引力模型）")
    print("=" * 60)
    print(f"初始位置: {r0} m")
    print(f"目标位置: {rf} m")
    print(f"飞行时间: {t_span[1]} s")
    
    results = {}
    times = {}
    
    # 1. 直接法
    from algorithms import DirectMethodOptimizer
    success, res, t = test_method("直接法",
        lambda: DirectMethodOptimizer(asteroid, spacecraft, n_nodes=15),
        r0, v0, m0, rf, vf, t_span)
    results['直接法'] = (success, res)
    times['直接法'] = t
    
    # 2. CVXPY凸优化
    from algorithms import CVXPYSCPOptimizer
    success, res, t = test_method("CVXPY凸优化",
        lambda: CVXPYSCPOptimizer(asteroid, spacecraft, n_nodes=15, max_iterations=50, verbose=False),
        r0, v0, m0, rf, vf, t_span)
    results['CVXPY凸优化'] = (success, res)
    times['CVXPY凸优化'] = t
    
    # 3. 快速伪谱法
    from algorithms import FastPseudospectralOptimizer
    success, res, t = test_method("快速伪谱法",
        lambda: FastPseudospectralOptimizer(asteroid, spacecraft, n_nodes=20, verbose=False),
        r0, v0, m0, rf, vf, t_span)
    results['快速伪谱法'] = (success, res)
    times['快速伪谱法'] = t
    
    # 4. 快速打靶法
    from algorithms import FastShootingOptimizer
    success, res, t = test_method("快速打靶法",
        lambda: FastShootingOptimizer(asteroid, spacecraft, n_nodes=20, verbose=False),
        r0, v0, m0, rf, vf, t_span)
    results['快速打靶法'] = (success, res)
    times['快速打靶法'] = t
    
    # 5. 快速同伦法
    from algorithms import FastHomotopyOptimizer
    success, res, t = test_method("快速同伦法",
        lambda: FastHomotopyOptimizer(asteroid, spacecraft, n_nodes=20, n_homotopy_steps=2, verbose=False),
        r0, v0, m0, rf, vf, t_span)
    results['快速同伦法'] = (success, res)
    times['快速同伦法'] = t
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"{'方法':<15} {'状态':<6} {'位置误差':<12} {'速度误差':<12} {'燃料(kg)':<10} {'耗时(s)':<8}")
    print("-" * 70)
    
    for name, (success, res) in results.items():
        if res:
            status = "成功" if success else "部分"
            print(f"{name:<15} {status:<6} {res['pos_err']:.4f}m{'':<4} {res['vel_err']:.4f}m/s{'':<4} {res['fuel']:<10.2f} {times[name]:<8.2f}")
        else:
            print(f"{name:<15} {'失败':<6} {'-':<12} {'-':<12} {'-':<10} {times[name]:<8.2f}")
    
    success_count = sum(1 for s, _ in results.values() if s)
    print(f"\n成功率: {success_count}/{len(results)} = {success_count/len(results)*100:.0f}%")
    
    # 找出最优方法
    best = None
    best_fuel = float('inf')
    for name, (success, res) in results.items():
        if success and res and res['fuel'] < best_fuel:
            best_fuel = res['fuel']
            best = name
    if best:
        print(f"\n燃料最优: {best} (燃料消耗: {best_fuel:.2f}kg)")
    
    # 速度最快
    fastest = min(times.items(), key=lambda x: x[1])
    print(f"速度最快: {fastest[0]} (计算时间: {fastest[1]:.2f}s)")


if __name__ == "__main__":
    main()