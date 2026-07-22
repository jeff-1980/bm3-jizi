# 嫁接预注册(物化自 vault 卡 bm3-graft-constructive,created 2026-07-03,早于本网格)
R = mean over {0,-2,-6dB} of (s4d_plus_gate - s4d) / (bm3_frozen - s4d)
- R >= 0.7  => CONSTRUCTIVE_CONFIRMED
- 0.3 <= R < 0.7 => PARTIAL_RECOVERY
- R < 0.3   => GRAFT_INSUFFICIENT
禁未预注册判据。基线 bm3_frozen/s4d 取 secondary_20260703-1034 同批值。
物化时间:网格起跑前。
