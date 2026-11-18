import sys
import traceback

print("Python version:", sys.version)
try:
    import pypfopt
    print("pypfopt version:", pypfopt.__version__)
    from pypfopt import EfficientFrontier, HRPOpt
    print("pypfopt imports OK")
except Exception as e:
    traceback.print_exc()

try:
    import src.stage4_backtest as s4
    print("Import stage4_backtest OK")
    s4.run_stage4(".")
except Exception as e:
    traceback.print_exc()
