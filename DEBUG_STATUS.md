== 2026-09-21 调试结论 ==
1. PP双卡数值正确: 双卡与单卡同flag首token逐位一致(97237)
2. 边界交换逐位一致(rank-out==rank-in)
3. 唯一坏组合: eager(NoGraph)+specNone — 原引擎从未被生产使用的角落(生产=MTP+graphs)
4. 下一步(三选一): A.图分rank(推荐,绕开bug+恢复解码性能) B.修eager ordinary C.PP仅MTP
5. 生产实例已全部恢复: 7004(vllm原参数)/7005/7006
