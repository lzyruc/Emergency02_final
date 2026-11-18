def remove_chair(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    #"""清理椅子：接近、双臂抓持、拖动、移出通道并回位。"""
    # 1. 根据target["pose"]，移动到1米前方
    # 2. 头部视觉识别椅子，给出新的pose
    # 3. 移动到0.4米前方
    # 4. 切换到预抓取姿态poses_chair/应急_转椅_背_左右侧.json
    # 5. 双腕部相机视觉识别抓取点，给出抓取点1 / 2 （waist_yaw_link坐标系下）
    # 6. 先选近的抓取点，如果小于40cm，左臂or右臂执行抓取，完成后不切换
    # 7. 如果抓取点均大于40cm，报失败（调试用），返回False
    # 8. 抓取后，turnTo(0)，moveTo("startpoint",correct=True,rotate=False), turnTo(90),moveBy(0.5,0,0) 松夹爪, recoverState("startpoint","move_by"),turnTo(0)，返回True
    return True

def remove_bin(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理垃圾桶：接近、预备、夹持、少量抬起、移出通道、释放并回位。"""
    # 待接入：垃圾桶三段预设姿态和经用户确认的放置路线。
    # 1. 根据target["pose"]，移动到1米前方
    # 2. 头部视觉识别垃圾桶，给出新的pose
    # 3. 移动到0.4米前方
    # 4. 切换到预抓取姿态poses_bin/应急_垃圾桶_1远观察.json
    # 5. 双腕部相机视觉识别垃圾桶，给出抓取点1 / 2 （waist_yaw_link坐标系下）
    # 6. 先选近的抓取点，如果小于40cm，左臂or右臂执行抓取，完成后切换到poses_bin/应急_垃圾桶_3提起.json
    # 7. 如果抓取点均大于40cm，报失败（调试用），返回False
    # 8. 抓取后，turnTo(0)，moveTo("startpoint",correct=True,rotate=False), turnTo(90),moveBy(0.5,0,0) 松夹爪, recoverState("startpoint","move_by"),turnTo(0)，返回True
    return True

def remove_box(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理 3 kg 纸箱：接近、双臂抱持、拖动、移出通道并回位。"""
    # 待接入：横箱/竖箱选择条件、三段预设姿态和放置路线。
    # 1. 根据target["pose"]，移动到1米前方
    # 2. 头部视觉识别纸箱，给出新的pose
    # 3. 移动到0.4米前方
    # 4. 切换到预抓取姿态poses_box/应急_纸箱_1预备.json
    # 5. 切换到poses_box/应急_纸箱_2横箱子.json
    # 6. TurnTo(270), moveTo("startpoint",correct=True,rotate=False), turnTo(0), moveBy(0,-0.3,0), turnTo(90),moveBy(0.8,0,0), recoverState("startpoint","move_by"), turnTo(0)，返回True
    return True


def remove_filebox(
    station: Any,
    navigation: Any,
    target: dict[str, Any],
) -> bool:
    """清理文件筐：两段视觉接近、腕部精定位、抓取、移出通道并回位。"""
    # 待接入：复用 blue_grasp，确认文件筐新旧预设姿态和放置路线。
    # 1. 根据target["pose"]，移动到1米前方
    # 2. 头部视觉识别文件筐，给出新的pose
    # 3. 移动到0.4米前方
    # 4. 切换到预抓取姿态poses_filebox/应急_文件框0.json
    # 5. 双腕部相机视觉识别文件筐，给出抓取点1 / 2 （waist_yaw_link坐标系下）
    # 6. 先选近的抓取点，如果小于40cm，左臂or右臂执行抓取，完成后切换到poses_filebox/应急_文件框_拿起.json
    # 7. 如果抓取点均大于40cm，切换到预抓取姿态poses_filebox/应急_文件框1.json
    # 8. 双腕部相机视觉识别文件筐，给出抓取点3 / 4
    # 9. 先选近的抓取点，如果小于40cm，左臂or右臂执行抓取，完成后切换到poses_filebox/应急_文件框_拿起.json
    # (可能失败怎么办)
    # 10. 如果仍然大于40cm，切换到预抓取姿态poses_filebox/应急_文件框2.json
    # 11. 双腕部相机视觉识别文件筐，给出抓取点5 / 6
    # 12. 先选近的抓取点，如果小于40cm，左臂or右臂执行抓取，完成后切换到poses_filebox/应急_文件框_拿起.json
    # 13. 如果仍然大于40cm，报失败（调试用），返回False
    # 14. 抓取后，turnTo(0)，moveTo("startpoint",correct=True,rotate=False), turnTo(90),moveBy(0.5,0,0) 松夹爪, recoverState("startpoint","move_by"),turnTo(0)，返回True
    return True
