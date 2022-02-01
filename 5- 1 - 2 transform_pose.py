import pyzed.sl as sl

print(sl.Transform())

def transform_pose(pose, tx, ty, tz) :
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    transform_[0][3] = tx
    transform_[1][3] = ty
    transform_[2][3] = tz
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform_)
    transform_inv.inverse()
    pose = transform_inv * pose * transform_