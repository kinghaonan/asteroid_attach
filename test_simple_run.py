#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 使用点质量引力模型

绕过DNN模型，直接测试轨迹优化算法。
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SimpleAsteroid:
    """简化的点质量引力模型"""
    
    def __init__(self, mu=11.0, omega=None):
        self.mu = mu  # 引力常数 (m^3/s^2)
        self.omega = omega if omega is not None else np.array([0.0, 0.0, 3.6e-4])
    
    def compute_gravity(self, position):
        """计算引力加速度（点质量模型）"""
        r = np.linalg.norm(position)
        if r < 1e-10:
            return np.zeros(3)
        return -self.mu * position / (r**3)


class SimpleSpacecraft:
    """简化的航天器模型"""
    
    def __init__(self):
        self.T_max = 20.0  # 最大推力 (N)
        self.I_sp = 300.0  # 比冲 (s)
        self.g0 = 9.80665  # 重力加速度 (m/s^2)
        self.m0 = 500.0    # 初始质量 (kg)


def main():
    print("="*70)
    print("简单轨迹优化测试（点质量引力模型）")
    print("="*70)
    
    # 创建模型
    asteroid = SimpleAsteroid(mu=11.0)
    spacecraft = SimpleSpacecraft()
    
    # 问题设置
    r0 = np.array([500.0, 0.0, 100.0])
    v0 = np.array([0.0, 0.2, 0.0])
    m0 = spacecraft.m0
    rf = np.array([400.0, 50.0, 50.0])
    vf = np.array([0.0, 0.1, 0.0])
    t_span = [0.0, 1800.0]
    
    print(f"\n问题设置:")
    print(f"  初始位置: {r0} m")
    print(f"  初始速度: {v0} m/s")
    print(f"  初始质量: {m0} kg")
    print(f"  目标位置: {rf} m")
    print(f"  目标速度: {vf} m/s")
    print(f"  飞行时间: {t_span[1]} s")
    
    # 延迟导入优化器
    from algorithms import (
        FastPseudospectralOptimizer,
        FastShootingOptimizer,
        FastHomotopyOptimizer,
    )
    
    methods = [
        ("伪谱法", FastPseudospectralOptimizer, {'n_nodes': 20, 'verbose': True}),
        ("打靶法", FastShootingOptimizer, {'n_nodes': 20, 'verbose': True}),
        ("同伦法", FastHomotopyOptimizer, {'n_nodes': 20, 'n_homotopy_steps': 2, 'verbose': True}),
    ]
    
    results = []
    
    for name, OptimizerClass, kwargs in methods:
        print(f"\n{'='*70}")
        print(f"测试方法: {name}")
        print("="*70)
        
        try:
            optimizer = OptimizerClass(asteroid, spacecraft, **kwargs)
            
            start_time = time.time()
            result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
            elapsed = time.time() - start_time
            
            if result and 'r' in result:
                pos_err = np.linalg.norm(result['r'][-1] - rf)
                vel_err = np.linalg.norm(result['v'][-1] - vf)
                fuel = m0 - result['m'][-1]
                success = pos_err < 10 and vel_err < 5
                
                print(f"\n结果:")
                print(f"  成功: {success}")
                print(f"  位置误差: {pos_err:.4f} m")
                print(f"  速度误差: {vel_err:.4f} m/s")
                print(f"  燃料消耗: {fuel:.2f} kg")
                print(f"  计算时间: {elapsed:.2f} s")
                
                results.append({
                    'method': name,
                    'success': success,
                    'pos_err': pos_err,
                    'vel_err': vel_err,
                    'fuel': fuel,
                    'time': elapsed,
                })
            else:
                print("  优化失败")
                results.append({'method': name, 'success': False})
                
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results.append({'method': name, 'success': False, 'error': str(e)})
    
    # 汇总结果
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    print(f"{'方法':<10} {'成功':<6} {'位置误差(m)':<12} {'速度误差(m/s)':<14} {'燃料(kg)':<10} {'时间(s)':<8}")
    print("-"*70)
    
    for r in results:
        if r['success']:
            print(f"{r['method']:<10} {'是':<6} {r['pos_err']:<12.4f} {r['vel_err']:<14.4f} {r['fuel']:<10.2f} {r['time']:<8.2f}")
        else:
            err_msg = r.get('error', '优化失败')
            print(f"{r['method']:<10} {'否':<6} {err_msg[:30]}")
    
    # 计算成功率
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n成功率: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")


if __name__ == "__main__":
    main()