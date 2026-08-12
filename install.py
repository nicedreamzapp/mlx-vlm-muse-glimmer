"""Copy the muse_glimmer package into the active mlx_vlm install."""
import os, shutil, mlx_vlm

dst = os.path.join(os.path.dirname(mlx_vlm.__file__), "models", "muse_glimmer")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(os.path.dirname(__file__), "muse_glimmer"), dst)
print("installed ->", dst)
