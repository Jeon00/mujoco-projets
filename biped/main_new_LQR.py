import mujoco as mj
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import scipy.linalg
import numpy as np
import os

xml_path = 'KAsimov_Humanoid_symmetry_leg.xml'
simend = 30

step_no = 0;

fsm_leg1_swing = 0 # states에 해당하는 변수들 선언
fsm_leg2_swing = 1

fsm_knee1_stance = 0
fsm_knee1_retract = 1
fsm_knee1_kick = 2

fsm_knee2_stance = 0
fsm_knee2_retract = 1
fsm_knee2_kick = 2

fsm_hip = fsm_leg2_swing # 초기 상태에 맞게 변수 지정
fsm_knee1 = fsm_knee1_stance
fsm_knee2 = fsm_knee2_stance

x_joint = 0
z_joint = 1
torso_joint = 11
right_thigh_joint = 13
right_knee_joint = 14
right_foot_joint_1 = 15
left_thigh_joint = 8
left_knee_joint = 9
left_foot_joint_1 = 10

hip1_pservo_y = 0
hip1_vservo_y = 1
hip2_pservo_y = 2
hip2_vservo_y = 3
knee1_pservo_y = 4
knee1_vservo_y = 5
knee2_pservo_y = 6
knee2_vservo_y = 7
anckle1_pservo_y = 8
anckle1_vservo_y = 9
anckle2_pservo_y = 10
anckle2_vservo_y = 11
anckle1_pservo_x = 12
anckle1_vservo_x = 13
anckle2_pservo_x = 14
anckle2_vservo_x = 15



world_body = 0
ass_body = 1
leg11_body = 2
leg12_body = 3
r_leg_link_5_1 = 11
leg21_body = 5
leg22_body = 6
l_leg_link_5_1 = 6

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

retract=0
kick=0
walk=0

def LQR(model, data):
    global dq, qpos0, ctrl0, K

    # mj.mj_resetDataKeyframe(model, data, 1)
    # mj.mj_forward(model, data)
    # data.qacc = 0  # Assert that there is no the acceleration.
    # mj.mj_inverse(model, data)
    # print(data.qfrc_inverse)

    # height_offsets = np.linspace(-0.001, 0.001, 2001)
    # vertical_forces = []
    # for offset in height_offsets:
    #     mj.mj_resetDataKeyframe(model, data, 1)
    #     mj.mj_forward(model, data)
    #     data.qacc = 0
    #     # Offset the height by `offset`.
    #     data.qpos[2] += offset
    #     mj.mj_inverse(model, data)
    #     vertical_forces.append(data.qfrc_inverse[2])

    # # Find the height-offset at which the vertical force is smallest.
    # idx = np.argmin(np.abs(vertical_forces))
    # best_offset = height_offsets[idx]

    # # Plot the relationship.
    # plt.figure(figsize=(10, 6))
    # plt.plot(height_offsets * 1000, vertical_forces, linewidth=3)
    # # Red vertical line at offset corresponding to smallest vertical force.
    # plt.axvline(x=best_offset*1000, color='red', linestyle='--')
    # # Green horizontal line at the humanoid's weight.
    # weight = model.body_subtreemass[1]*np.linalg.norm(model.opt.gravity)
    # plt.axhline(y=weight, color='green', linestyle='--')
    # plt.xlabel('Height offset (mm)')
    # plt.ylabel('Vertical force (N)')
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # plt.minorticks_on()
    # plt.title(f'Smallest vertical force '
    #         f'found at offset {best_offset*1000:.4f}mm.')
    # plt.show()

    # mj.mj_resetDataKeyframe(model, data, 1)
    # mj.mj_forward(model, data)
    # data.qacc = 0
    # data.qpos[2] += best_offset
    qpos0 = data.qpos.copy()  # Save the position setpoint.
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    # print('desired forces:', qfrc0)

    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    # print('control setpoint:', ctrl0)

    data.ctrl = ctrl0
    # mj.mj_forward(model, data)
    # print('actuator forces:', data.qfrc_actuator)


    nu = model.nu  # Alias for the number of actuators.
    nv=model.nv # Shortcut for the number of DoFs.
        

    # Get the Jacobian for the root body (torso) CoM.
    mj.mj_resetData(model, data)
    data.qpos = qpos0
    # mj.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    mj.mj_jacSubtreeCom(model, data, jac_com, model.body('base_link').id)

    # Get the Jacobian for the left foot.
    jac_foot = np.zeros((3, nv))
    mj.mj_jacBodyCom(model, data, jac_foot, None, model.body('l_leg_link_5_1').id)

    jac_diff = jac_com - jac_foot
    Qbalance = jac_diff.T @ jac_diff

    # Get all joint names.
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    # Get indices into relevant sets of joints.
    root_dofs = range(6)
    body_dofs = range(6, nv)
    abdomen_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'torso' in name
    ]
    left_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'left' in name
        and ('pelvis' in name or 'thigh' in name or 'knee' in name or 'foot' in name)
    ]
    balance_dofs = abdomen_dofs + left_leg_dofs
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)



    ### Choose the way to set Q and R ###

    QR_setting_value=2 # 1 for simple way(from double_pendulum_lqr) 
                        # 2 for elaborate way(from humanoid_LQR_example)

    if QR_setting_value==1:
        #### Example adjustment of Q and R matrices
        Q = np.eye(2 * nv)  # Modify based on your system dynamics
        R = np.eye(nu) * 1e-2  # Modify based on your control input
    
    else:
        R = np.eye(nu)
        # # Cost coefficients.
        BALANCE_COST        = 1000  # Balancing.
        BALANCE_JOINT_COST  = 2    # Joints required for balancing.
        OTHER_JOINT_COST    = 0.3   # Other joints.

        # Construct the Qjoint matrix.
        Qjoint = np.eye(nv)
        Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
        Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

        # Construct the Q matrix for position DoFs.
        Qpos = BALANCE_COST * Qbalance + Qjoint

        # No explicit penalty for velocities.
        Q = np.block([[Qpos, np.zeros((nv, nv))],
                    [np.zeros((nv, 2*nv))]])
        
    # Set the initial state and control.
    mj.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    # Allocate the A and B matrices, compute them.
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # print(K)

    # Reset data, set initial pose.
    mj.mj_resetData(model, data)
    data.qpos = qpos0

    # Allocate position difference dq.
    dq = np.zeros(model.nv)

def LQR_middle(model, data):
    global dq, qpos0, ctrl0, K

    # mj.mj_resetDataKeyframe(model, data, 1)
    # mj.mj_forward(model, data)
    # data.qacc = 0  # Assert that there is no the acceleration.
    # mj.mj_inverse(model, data)
    # print(data.qfrc_inverse)

    # height_offsets = np.linspace(-0.001, 0.001, 2001)
    # vertical_forces = []
    # for offset in height_offsets:
    #     mj.mj_resetDataKeyframe(model, data, 1)
    #     mj.mj_forward(model, data)
    #     data.qacc = 0
    #     # Offset the height by `offset`.
    #     data.qpos[2] += offset
    #     mj.mj_inverse(model, data)
    #     vertical_forces.append(data.qfrc_inverse[2])

    # # Find the height-offset at which the vertical force is smallest.
    # idx = np.argmin(np.abs(vertical_forces))
    # best_offset = height_offsets[idx]

    # # Plot the relationship.
    # plt.figure(figsize=(10, 6))
    # plt.plot(height_offsets * 1000, vertical_forces, linewidth=3)
    # # Red vertical line at offset corresponding to smallest vertical force.
    # plt.axvline(x=best_offset*1000, color='red', linestyle='--')
    # # Green horizontal line at the humanoid's weight.
    # weight = model.body_subtreemass[1]*np.linalg.norm(model.opt.gravity)
    # plt.axhline(y=weight, color='green', linestyle='--')
    # plt.xlabel('Height offset (mm)')
    # plt.ylabel('Vertical force (N)')
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # plt.minorticks_on()
    # plt.title(f'Smallest vertical force '
    #         f'found at offset {best_offset*1000:.4f}mm.')
    # plt.show()

    # mj.mj_resetDataKeyframe(model, data, 1)
    # mj.mj_forward(model, data)
    # data.qacc = 0
    # data.qpos[2] += best_offset
    qpos0 = data.qpos.copy()  # Save the position setpoint.
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    # print('desired forces:', qfrc0)

    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    # print('control setpoint:', ctrl0)

    data.ctrl = ctrl0
    # mj.mj_forward(model, data)
    # print('actuator forces:', data.qfrc_actuator)


    nu = model.nu  # Alias for the number of actuators.
    nv=model.nv # Shortcut for the number of DoFs.
        

    # Get the Jacobian for the root body (torso) CoM.
    mj.mj_resetData(model, data)
    data.qpos = qpos0
    # mj.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    mj.mj_jacSubtreeCom(model, data, jac_com, model.body('base_link').id)

    # Get the Jacobian for the left foot.
    jac_foot_left = np.zeros((3, nv))
    mj.mj_jacBodyCom(model, data, jac_foot_left, None, model.body('l_leg_link_5_1').id)
   
    # Get the Jacobian for the right foot.
    jac_foot_right = np.zeros((3, nv))
    mj.mj_jacBodyCom(model, data, jac_foot_right, None, model.body('r_leg_link_5_1').id)

    jac_foot_middle=(jac_foot_left+jac_foot_right)/2


    jac_diff = jac_com - jac_foot_middle
    Qbalance = jac_diff.T @ jac_diff

    # Get all joint names.
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    # Get indices into relevant sets of joints.
    root_dofs = range(6)
    body_dofs = range(6, nv)
    abdomen_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'torso' in name
    ]
    left_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'left' in name
        and ('pelvis' in name or 'thigh' in name or 'knee' in name or 'foot' in name)
    ]
    right_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'right' in name
        and ('pelvis' in name or 'thigh' in name or 'knee' in name or 'foot' in name)
    ]

    balance_dofs = abdomen_dofs + left_leg_dofs + right_leg_dofs
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)



    ### Choose the way to set Q and R ###

    QR_setting_value=2 # 1 for simple way(from double_pendulum_lqr) 
                        # 2 for elaborate way(from humanoid_LQR_example)

    if QR_setting_value==1:
        #### Example adjustment of Q and R matrices
        Q = np.eye(2 * nv)  # Modify based on your system dynamics
        R = np.eye(nu) * 1e-2  # Modify based on your control input
    
    else:
        R = np.eye(nu)
        # # Cost coefficients.
        BALANCE_COST        = 1000  # Balancing.
        BALANCE_JOINT_COST  = 2    # Joints required for balancing.
        OTHER_JOINT_COST    = 0.3   # Other joints.

        # Construct the Qjoint matrix.
        Qjoint = np.eye(nv)
        Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
        Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

        # Construct the Q matrix for position DoFs.
        Qpos = BALANCE_COST * Qbalance + Qjoint

        # No explicit penalty for velocities.
        Q = np.block([[Qpos, np.zeros((nv, nv))],
                    [np.zeros((nv, 2*nv))]])
        
    # Set the initial state and control.
    mj.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    # Allocate the A and B matrices, compute them.
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # print(K)

    # Reset data, set initial pose.
    mj.mj_resetData(model, data)
    data.qpos = qpos0

    # Allocate position difference dq.
    dq = np.zeros(model.nv)


def controller(model, data):
    """
    This function implements a controller that
    mimics the forces of a fixed joint before release
    """ 
    global fsm_hip
    global fsm_knee1
    global fsm_knee2
    global step_no

    # 변수 정의
    l_1 = 0.07450
    l_2 = 0.07705
    l_3 = 0.1  # 다리 길이
    l_stance = 0.140  # 서있을 때 펴고 있을 다리 길이
    theta14, theta14dot = 0, 0  # 허리->발목 벡터와 몸통 z축 사잇각 for leg1
    theta24, theta24dot = 0, 0
    abs_theta_leg1, abs_theta_leg2 = 0, 0  # world 좌표계에서 다리 각도
    l_14, l_24 = 0, 0  # 허리->발목 길이
    z_foot1, z_foot2 = 0, 0

    theta11_ctrl, theta12_ctrl, theta13_ctrl = 0, 0, 0  # 지정할 다리 부분 조인트각도
    theta21_ctrl, theta22_ctrl, theta23_ctrl = 0, 0, 0
    l_14_ctrl, l_24_ctrl = 0, 0  # 지정할 허리->발목 길이
    theta14_ctrl, theta24_ctrl = 0, 0  # 지정할 허리->발목 각도

    kick_dis = 0.007  # kick 할 정도 결정
    z_foot_kickStop = 0.06  # 킥 모션을 중지할 발 높이

    retract_dis = 0.006

    global retract, kick, walk

    # get position and vel of joints
    x = data.qpos[x_joint]
    vx = data.qvel[x_joint]
    z = data.qpos[z_joint]
    vz = data.qvel[z_joint]
    theta0 = data.qpos[torso_joint]
    theta0dot = data.qvel[torso_joint]
    theta11 = data.qpos[right_thigh_joint]
    theta11dot = data.qvel[right_thigh_joint]
    theta21 = data.qpos[left_thigh_joint]
    theta21dot = data.qvel[left_thigh_joint]
    theta12 = data.qpos[right_knee_joint]
    theta12dot = data.qvel[right_knee_joint]
    theta22 = data.qpos[left_knee_joint]
    theta22dot = data.qvel[left_knee_joint]
    theta13 = data.qpos[right_foot_joint_1]
    theta13dot = data.qvel[right_foot_joint_1]
    theta23 = data.qpos[left_foot_joint_1]
    theta23dot = data.qvel[left_foot_joint_1]

    base_angle_y=quat2euler(data.xquat[1])[1]
    base_angle_x=quat2euler(data.xquat[1]) [0]
    # print(base_angle_x)
    # print(data.qpos[11], data.qpos[16])


    # State Estimation
    joint_no1 = right_thigh_joint
    joint_no2 = right_foot_joint_1

    # l_14 계산
    l_14 = get_len_4(data.xanchor[joint_no1, 0], data.xanchor[joint_no1, 1], data.xanchor[joint_no1, 2], data.xanchor[joint_no2, 0], data.xanchor[joint_no2, 1], data.xanchor[joint_no2, 2])
    
    joint_no1 = left_thigh_joint
    joint_no2 = left_foot_joint_1
    l_24 = get_len_4(data.xanchor[joint_no1, 0], data.xanchor[joint_no1, 1], data.xanchor[joint_no1, 2], data.xanchor[joint_no2, 0], data.xanchor[joint_no2, 1], data.xanchor[joint_no2, 2])

    abs_theta_leg1 = get_theta_n4(l_1, l_2, theta11, theta12)
    # abs_theta_leg1 = theta0+theta14

    abs_theta_leg2 = get_theta_n4(l_1, l_2, theta21, theta22)
    # abs_theta_leg2 = theta0+theta24;

    #position of foot1
    body_no = r_leg_link_5_1
    #x = d->xpos[3*body_no]; y = d->qpos[3*body_no+1]; 
    z_foot1 = data.xpos[body_no, 2]+0.392747
    #printf("%f \n", z_foot1);
  
    body_no = l_leg_link_5_1
    z_foot2 = data.xpos[body_no, 2]+0.392747

    # print(z_foot1, z_foot2)
    # print(fsm_hip, fsm_knee1, fsm_knee2, counter)
    # print(data.xanchor[right_foot_joint_1, 0]<data.xanchor[right_thigh_joint, 0])
    # print(base_angle)
    # print(counter)

    # quat_leg1 = data.xquat[1, :] # cartesian orientation of body frame
    # euler_leg1 = quat2euler(quat_leg1)
    # abs_leg1 = -euler_leg1[1]
    # pos_foot1 = data.xpos[2, :]

    # quat_leg2 = data.xquat[3, :]
    # euler_leg2 = quat2euler(quat_leg2)
    # abs_leg2 = -euler_leg2[1]
    # pos_foot2 = data.xpos[4, :]

    dis_set=1
  
    dis_left_thigh_x=data.xanchor[left_thigh_joint, 0]
    dis_left_foot_x=data.xanchor[left_foot_joint_1, 0]
    dis_right_thigh_x=data.xanchor[right_thigh_joint, 0]
    dis_right_foot_x=data.xanchor[right_foot_joint_1, 0]
    dis_diff=0.01765

    dis_left_thigh=cal_dis(data.xanchor[torso_joint, 0], data.xanchor[torso_joint, 1], data.xanchor[left_thigh_joint, 0], data.xanchor[left_thigh_joint, 1])
    dis_left_foot=cal_dis(data.xanchor[torso_joint, 0], data.xanchor[torso_joint, 1], data.xanchor[left_foot_joint_1, 0], data.xanchor[left_foot_joint_1, 1])
    dis_right_thigh=cal_dis(data.xanchor[torso_joint, 0], data.xanchor[torso_joint, 1], data.xanchor[right_thigh_joint, 0], data.xanchor[right_thigh_joint, 1])
    dis_right_foot=cal_dis(data.xanchor[torso_joint, 0], data.xanchor[torso_joint, 1], data.xanchor[right_foot_joint_1, 0], data.xanchor[right_foot_joint_1, 1])
    # print(z_foot2, dis_right_foot, dis_right_thigh)
    print(fsm_hip, fsm_knee1, fsm_knee2, "\\", retract, kick, "\\", walk)

    # Transition check
    if True:  # 그냥 코드 접으려고 if문 추가
        if dis_set==0:
            if fsm_hip == fsm_leg2_swing and z_foot2 < 0.03 and dis_right_foot_x-dis_diff<dis_right_thigh_x:
                fsm_hip = fsm_leg1_swing
                walk+=1

            if fsm_hip == fsm_leg1_swing and z_foot1 < 0.03 and dis_left_foot_x-dis_diff<dis_left_thigh_x:
                fsm_hip = fsm_leg2_swing
                walk+=1

            if fsm_hip == fsm_leg2_swing and fsm_knee1 == fsm_knee1_stance and z_foot2 < 0.036 and dis_right_foot_x-dis_diff<dis_right_thigh_x:  # kick state for leg1
                fsm_knee1 = fsm_knee1_kick
                kick+=1

            if fsm_knee1 == fsm_knee1_kick and z_foot1 > z_foot_kickStop and dis_right_foot_x-dis_diff<dis_right_thigh_x:  # modified retract state for leg1
                fsm_knee1 = fsm_knee1_retract
                retract+=1

            if fsm_knee1 == fsm_knee1_retract and dis_right_foot_x-dis_diff>dis_right_thigh_x:
                fsm_knee1 = fsm_knee1_stance

            if fsm_hip == fsm_leg1_swing and fsm_knee2 == fsm_knee2_stance and z_foot1 < 0.036 and dis_left_foot_x-dis_diff<dis_left_thigh_x:  # kick state for leg2
                fsm_knee2 = fsm_knee2_kick
                kick+=1

            if fsm_knee2 == fsm_knee2_kick and z_foot2 > z_foot_kickStop and dis_left_foot_x-dis_diff<dis_left_thigh_x:  # modified retract state for leg2
                fsm_knee2 = fsm_knee2_retract
                retract+=1

            if fsm_knee2 == fsm_knee2_retract and dis_left_foot_x-dis_diff>dis_left_thigh_x:
                    fsm_knee2 = fsm_knee2_stance
        
        if dis_set==1:

            if fsm_hip == fsm_leg2_swing and z_foot2 < 0.03 and dis_right_foot-dis_diff<dis_right_thigh:
                fsm_hip = fsm_leg1_swing
                walk+=1

            if fsm_hip == fsm_leg1_swing and z_foot1 < 0.03 and dis_left_foot-dis_diff<dis_left_thigh:
                fsm_hip = fsm_leg2_swing
                walk+=1

            if fsm_hip == fsm_leg2_swing and fsm_knee1 == fsm_knee1_stance and z_foot2 < 0.036 and dis_right_foot-dis_diff<dis_right_thigh:  # kick state for leg1
                fsm_knee1 = fsm_knee1_kick
                kick+=1

            if fsm_knee1 == fsm_knee1_kick and z_foot1 > z_foot_kickStop and dis_right_foot-dis_diff<dis_right_thigh:  # modified retract state for leg1
                fsm_knee1 = fsm_knee1_retract
                retract+=1

            if fsm_knee1 == fsm_knee1_retract and dis_right_foot-dis_diff>dis_right_thigh:
                fsm_knee1 = fsm_knee1_stance

            if fsm_hip == fsm_leg1_swing and fsm_knee2 == fsm_knee2_stance and z_foot1 < 0.036 and dis_left_foot-dis_diff<dis_left_thigh:  # kick state for leg2
                fsm_knee2 = fsm_knee2_kick
                kick+=1

            if fsm_knee2 == fsm_knee2_kick and z_foot2 > z_foot_kickStop and dis_left_foot-dis_diff<dis_left_thigh:  # modified retract state for leg2
                fsm_knee2 = fsm_knee2_retract
                retract+=1

            if fsm_knee2 == fsm_knee2_retract and dis_left_foot-dis_diff>dis_left_thigh:
                    fsm_knee2 = fsm_knee2_stance

    # if fsm_hip == FSM_LEG2_SWING and pos_foot2[2] < 0.05 and abs_leg1 < 0.0: # foot2가 바닥에 닿아있고 leg1이 뒤에 있을 때
    #     fsm_hip = FSM_LEG1_SWING # leg1이 움직이도록
    # if fsm_hip == FSM_LEG1_SWING and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
    #     fsm_hip = FSM_LEG2_SWING

    # if fsm_knee1 == FSM_KNEE1_STANCE and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
    #     fsm_knee1 = FSM_KNEE1_RETRACT # 걷는 도중(foot2가 바닥에)에는 knee1이 들어감
    # if fsm_knee1 == FSM_KNEE1_RETRACT and abs_leg1 > 0.1:
    #     fsm_knee1 = FSM_KNEE1_STANCE # 걷고나서는 knee1이 원위치

    # if fsm_knee2 == FSM_KNEE2_STANCE and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
    #     fsm_knee2 = FSM_KNEE2_RETRACT
    # if fsm_knee2 == FSM_KNEE2_RETRACT and abs_leg2 > 0.1:
    #     fsm_knee2 = FSM_KNEE2_STANCE

    # # Control
    # if fsm_hip == FSM_LEG1_SWING:
    #     data.ctrl[0] = -0.5 # leg1이 움직이도록
    # if fsm_hip == FSM_LEG2_SWING:
    #     data.ctrl[0] = 0.5

    # if fsm_knee1 == FSM_KNEE1_STANCE:
    #     data.ctrl[2] = 0.0 # knee1이 원위치
    # if fsm_knee1 == FSM_KNEE1_RETRACT:
    #     data.ctrl[2] = -0.25 # knee1이 들어감

    # if fsm_knee2 == FSM_KNEE2_STANCE:
    #     data.ctrl[4] = 0.0
    # if fsm_knee2 == FSM_KNEE2_RETRACT:
    #     data.ctrl[4] = -0.25
                
    # All stabilizer here
    if True:
        # 여기다가 발바닥 각도 넣어주면 될듯.
        data.ctrl[anckle1_pservo_y] = -( -base_angle_y + theta11 + theta12)
        data.ctrl[anckle2_pservo_y] = -( -base_angle_y + theta21 + theta22)
        # data.ctrl[anckle1_pservo_x] = -base_angle_x
        # data.ctrl[anckle2_pservo_x] = -base_angle_x

        # if base_angle_x>0:

        #     data.ctrl[anckle1_pservo_x] = -base_angle_x+0.05
        #     data.ctrl[anckle2_pservo_x] = -base_angle_x+0.05
            
        # if base_angle_x<0:
        #     data.ctrl[anckle1_pservo_x] = -base_angle_x-0.05
        #     data.ctrl[anckle2_pservo_x] = -base_angle_x-0.05


        # All actions here
        if fsm_hip == fsm_leg1_swing:
            theta14_ctrl = 50*np.pi/180
            theta24_ctrl = -40*np.pi/180

        if fsm_hip == fsm_leg2_swing:
            theta14_ctrl = -40*np.pi/180
            theta24_ctrl = 50*np.pi/180

        if fsm_knee1 == fsm_knee1_stance:
            l_14_ctrl = l_stance

        if fsm_knee1 == fsm_knee1_kick:
            l_14_ctrl = l_stance + kick_dis

        if fsm_knee1 == fsm_knee1_retract:
            l_14_ctrl = l_stance - retract_dis

        if fsm_knee2 == fsm_knee2_stance:
            l_24_ctrl = l_stance

        if fsm_knee2 == fsm_knee2_kick:
            l_24_ctrl = l_stance + kick_dis

        if fsm_knee2 == fsm_knee2_retract:
            l_24_ctrl = l_stance - retract_dis

        # Action for leg 1
        theta11_ctrl, theta12_ctrl= get_leg_ctrl_radian(l_1, l_2, l_14_ctrl, theta14_ctrl)
        data.ctrl[hip1_pservo_y] = theta11_ctrl
        data.ctrl[knee1_pservo_y] = theta12_ctrl

        # Action for leg 2
        theta11_ctrl, theta12_ctrl = get_leg_ctrl_radian(l_1, l_2, l_24_ctrl, theta24_ctrl)
        data.ctrl[hip2_pservo_y] = theta11_ctrl
        data.ctrl[knee2_pservo_y] = theta12_ctrl

        # LQR(model, data)


def init_controller(model,data):
    # data.qpos[4] = 0.5 # hip joint(leg2가 우선 움직임)
    # data.ctrl[0] = data.qpos[4] # pservo of hip joint

    left=1
    if left==True:
    
        init_l1 = 0.1326
        init_l2 = 0.1326
        r_tmp_theta1, r_tmp_theta2 = get_leg_ctrl_radian(0.07450,0.07705,init_l1,0)
        # print(r_tmp_theta1, r_tmp_theta2)
        data.ctrl[hip1_pservo_y] = r_tmp_theta1
        data.ctrl[knee1_pservo_y] = r_tmp_theta2
        data.ctrl[anckle1_pservo_y] = 0

        tmp_theta1, tmp_theta2 = get_leg_ctrl_radian(0.07450,0.07705,init_l2, 0.5)
        # print(tmp_theta1, tmp_theta2)
        data.ctrl[hip2_pservo_y] = tmp_theta1
        data.ctrl[knee2_pservo_y] = tmp_theta2 #왼발 앞으로
        data.ctrl[anckle2_pservo_y] = -(tmp_theta1+tmp_theta2) #왼발바닥 
        
        # data.qpos[left_thigh_joint]=0.2
        # data.qpos[left_knee_joint]=-0.2
        # # data.qpos[left_foot_joint_1]=-0.2

        # data.qpos[right_thigh_joint]=-0.2
        # data.qpos[right_knee_joint]=0.2
        # # data.qpos[right_foot_joint_1]=0.2
    

        fsm_hip = fsm_leg2_swing
        fsm_knee1 = fsm_knee1_stance
        fsm_knee2 = fsm_knee2_stance
    
    else:


        init_l1 = 0.1326
        init_l2 = 0.1326
        r_tmp_theta1, r_tmp_theta2 = get_leg_ctrl_radian(0.07450,0.07705,init_l1,0.5)
        data.ctrl[hip1_pservo_y] = r_tmp_theta1
        data.ctrl[knee1_pservo_y] = r_tmp_theta2 #오른발 앞으로
        data.ctrl[anckle1_pservo_y] =  -(r_tmp_theta1+r_tmp_theta2)

        tmp_theta1, tmp_theta2 = get_leg_ctrl_radian(0.07450,0.07705,init_l2, 0)
        data.ctrl[hip2_pservo_y] = tmp_theta1
        data.ctrl[knee2_pservo_y] = tmp_theta2 
        data.ctrl[anckle2_pservo_y] = 0
        fsm_hip = fsm_leg1_swing
        fsm_knee1 = fsm_knee1_stance
        fsm_knee2 = fsm_knee2_stance

    # print(data.xpos[r_leg_link_5_1, 2]+0.332747)
    # print(data.xpos[l_leg_link_5_1, 2]+0.332747)



def quat2euler(quat):
    # SciPy defines quaternion as [x, y, z, w]
    # MuJoCo defines quaternion as [w, x, y, z]
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)

    # roll-pitch-yaw is the same as rotating w.r.t
    # the x, y, z axis in the world frame
    euler = r.as_euler('xyz', degrees=False)

    return euler
import math

# def get_leg_ctrl_radian(l_1, l_2, l_ctrl, theta_input):
#     # Link lengths
#     a = l_1
#     b = l_2

#     # End effector position
#     x = l_ctrl * math.cos(theta_input)
#     y = l_ctrl * math.sin(theta_input)

#     # Calculate theta_2
#     D = (x**2 + y**2 - a**2 - b**2) / (2 * a * b)
#     theta_2 = math.atan2(-math.sqrt(1 - D**2), D)

#     # Calculate theta_1
#     theta_1 = math.atan2(y, x) - math.atan2(b * math.sin(theta_2), a + b * math.cos(theta_2))

#     # Return results
#     return theta_1, theta_2

def get_leg_ctrl_radian(l_1, l_2, l_ctrl, theta_input):
    theta_2 =  -math.acos((pow(l_ctrl,2)- pow(l_1,2)-pow(l_2,2))/(2*l_1*l_2))
    if (theta_input<0):
        theta_1 = theta_input+math.acos((pow(l_1,2)+pow(l_ctrl,2)-pow(l_2,2))/(2*l_1*l_ctrl));
    
    else:
        theta_1 = theta_input+math.acos((pow(l_1,2)+pow(l_ctrl,2)-pow(l_2,2))/(2*l_1*l_ctrl));
    
    # Return results
    return theta_1, theta_2

def get_len_4(x1, y1, z1, x2, y2, z2):
    tmp1 = x2 - x1
    tmp2 = y2 - y1
    tmp3 = z2 - z1
    return math.sqrt(pow(tmp1, 2) + pow(tmp2, 2) + pow(tmp3, 2))

def get_theta_n4(l1, l2, theta1, theta2):
    return math.atan2(l1 * math.sin(theta1) - l2 * math.sin(theta2 - theta1),
                      l1 * math.cos(theta1) + l2 * math.cos(theta2 - theta1))

def cal_dis(x1, y1, x2, y2):
    tmp1 = x2 - x1
    tmp2 = y2 - y1
    return math.sqrt(pow(tmp1, 2) + pow(tmp2, 2))



def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Set camera configuration
cam.azimuth = 120  # 89.608063
cam.elevation = -10  # -11.588379
cam.distance = 1.5  # 5.0
cam.lookat = np.array([0.0, 0.0, 0.0])

#turn the direction of gravity to simulate a ramp
gamma=0
gravity=9.81
model.opt.gravity[0] = gravity * np.sin(gamma) # downhill과 같이 중력 설정
model.opt.gravity[2] = -gravity * np.cos(gamma)

init_controller(model,data)
LQR_middle(model, data)

#set the controller
#mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        #simulation step
        mj.mj_step(model, data)
        # Apply control
        controller(model, data)
        # Get state difference dx.
        mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        # LQR control law.
        ctrl_values= ctrl0 - K @ dx
        data.ctrl[16:]=ctrl_values[16:]
        # print(ctrl_values[16:])
        # print(data.qpos[18])

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Show joint frames
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

    # Update scene and render
    cam.lookat[0] = data.qpos[0] #camera follows the robot
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
