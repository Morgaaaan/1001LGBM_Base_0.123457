import sys, os
print("=== DEBUG ENV ===")
print("PYEXE =", sys.executable)
print("CWD   =", os.getcwd())
try:
    import lightgbm
    print("LGBM  =", lightgbm.__file__, getattr(lightgbm, "__version__", ""))
except Exception as e:
    print("IMPORT ERR:", e)
    raise
print("=== END DEBUG ===")