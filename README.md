# pairwise-dualquat-skin

## 领域

游戏、图形与多媒体

## 主题

双四元数蒙皮

## 说明

运行时归一、索引保护与重建双缓冲。

初始状态下 `graphics/dualquat_skin.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
