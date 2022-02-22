from cv2 import transform
import pyzed.sl as sl
import numpy as np
import pandas as pd

transform_ = sl.Transform()
transform_.set_identity()

transform_inv = sl.Transform()
transform_inv.init_matrix(transform_)
transform_inv.inverse()

print(transform_)