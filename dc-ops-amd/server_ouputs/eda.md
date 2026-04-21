(.venv) root@7:~/dc_ops_training/dc-ops-amd# python scripts/eda.py 
/usr/lib/python3.12/importlib/__init__.py:90: UserWarning: A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.4.4)
  return _bootstrap._gcd_import(name[level:], package, level)

═══════ DC-Ops SFT dataset EDA ═══════
source: HF Melikshah/dc-ops-sft-data
raw episodes:          1,083
dropped VAR_* rows:    149  (13.8%)
kept episodes:         934
    A1:  145  (15.5%)
    A2:  225  (24.1%)
    A4:  181  (19.4%)
    B1:  120  (12.8%)
    B3:  160  (17.1%)
    B4:  103  (11.0%)

command frequency (GPT turns, kept episodes):
      3658  set_rack_load  (51.5%)
      1128  diagnose  (15.9%)
       853  check_status  (12.0%)
       650  adjust_setpoint  (9.1%)
       249  start_generator  (3.5%)
       197  wait  (2.8%)
       137  set_fan_speed  (1.9%)
       136  acknowledge_alarm  (1.9%)
        52  start_crac  (0.7%)
        25  set_ups_mode  (0.4%)
        14  stop_crac  (0.2%)
         8  stop_generator  (0.1%)

length stats (chars — rough token ≈ chars/4):
   full GPT turn        n= 7107  mean=2977.8  median= 2893  p95= 4409  p99= 4509  max=4771
   <think> block        n= 7107  mean=2541.3  median= 2463  p95= 3965  p99= 3995  max=4002
   <reasoning> block    n= 7107  mean= 356.6  median=  351  p95=  467  p99=  546  max=812
   <command> block      n= 7107  mean=  18.9  median=   20  p95=   25  p99=   27  max=36

→ max_seq_length recommendation (windowed prompts):
   window p99 (upper-bound estimate) ≈ 2584 tokens
   empirical p99 (tokenizer-measured): ~2700 tokens
   empirical max (tokenizer-measured): ~2900 tokens
   configs/sft.yaml sets max_seq_length=4096 → safe (covers max + packing headroom)
   ⚠  NEVER drop max_seq_length below 3072 — would truncate >0% of windows