# Results / 结果

`metrics.csv` contains all eight paper comparisons; only source code for the six controlled configurations is included. `predictions/` contains their 12 within-region prediction files. `splits/` and `station_selection/` preserve exact event membership/order and test station choices. Event IDs refer to original catalogue row indices. All numerical values are unchanged from the verified final paper results.

总表保留八种方法对比，源码只提供六种受控结构；固定划分和台站选择为复现所需，不属于内部日志。所有指标数值保持不变。

`cross_region/` contains the six controlled models' archived transfer results. Alaska `no_message_attention` used the original pre-retry checkpoint in this transfer experiment, while the main within-region table uses the verified retry. Target-region coordinate bounds and the full target catalogue are used.

跨区结果不与区内测试混用；阿拉斯加无消息传递＋注意力池化的跨区记录仍来自修复前权重。原失败记录与内部核验报告另存于作者本地备份，不放进本包。
