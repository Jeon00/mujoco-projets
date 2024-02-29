import mujoco as mj
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

xml_path = 'KAsimov_Humanoid_biped_walking.xml'
simend = 30

step_no = 0

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

#joint indexing
virtual_joint = 0
left_pelvis_joint = 1
left_thigh_joint = 2
left_knee_joint = 3
left_foot_joint_1 = 4
left_foot_joint_2 = 5
right_pelvis_joint = 6
right_thigh_joint = 7
right_knee_joint = 8
right_foot_joint_1 = 9
right_foot_joint_2 = 10
torso_joint = 11
left_shoulder_joint_1 = 12
left_shoulder_joint_2 = 13
left_shoulder_joint_3 = 14
left_elbow_joint = 15
left_wrist_joint_1 = 16
left_wrist_joint_2 = 17
right_shoulder_joint_1 = 18
right_shoulder_joint_2 = 19
right_shoulder_joint_3 = 20
right_elbow_joint = 21
right_wrist_joint_1 = 22
right_wrist_joint_2 = 23
head_joint = 24

#actuator indexing
left_pelvis_joint_motor = 0
hip1_pservo_y = 1
hip1_vservo_y = 2
knee1_pservo_y = 3
knee1_vservo_y = 4
anckle1_pservo_y = 5
anckle1_vservo_y = 6
left_foot_joint_2_motor = 7
right_pelvis_joint_motor = 8
hip2_pservo_y = 9
hip2_vservo_y = 10
knee2_pservo_y = 11
knee2_vservo_y = 12
anckle2_pservo_y = 13
anckle2_vservo_y = 14
right_foot_joint_2_motor = 15
torso_joint_motor = 16
left_shoulder_joint_1_motor = 17
left_shoulder_joint_2_motor = 18
left_shoulder_joint_3_motor = 19
left_elbow_joint_motor = 20
left_wrist_joint_1_motor = 21
left_wrist_joint_2_motor = 22
head_joint_motor = 23
right_shoulder_joint_1_motor = 24
right_shoulder_joint_2_motor = 25
right_shoulder_joint_3_motor = 26
right_elbow_joint_motor = 27
right_wrist_joint_1_motor = 28
right_wrist_joint_2_motor = 29

#body indexing
base_link = 0
#ass_body = 1 #whole body
leg11_body = 7 #right thigh r_leg_link_2_1
leg12_body = 8 #right calf
foot1_body = 9 #right foot
leg21_body = 2 #left thigh
leg22_body = 3 #left calf
foot2_body = 4 #left foot

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

theta0 = 0

def controller(model, data):
    global fsm_hip
    global fsm_knee1
    global fsm_knee2
    global step_no

    print(fsm_hip, fsm_knee1, fsm_knee2)

    # 변수 정의
    l_1 = 0.19
    l_2 = 0.19
    l_stance = 0.15  # 서있을 때 펴고 있을 다리 길이
    abs_theta_leg1, abs_theta_leg2 = 0, 0  # world 좌표계에서 다리 각도
    z_foot1, z_foot2 = 0, 0

    theta11_ctrl, theta12_ctrl, theta13_ctrl = 0, 0, 0  # 지정할 다리 부분 조인트각도
    theta21_ctrl, theta22_ctrl, theta23_ctrl = 0, 0, 0
    l_14_ctrl, l_24_ctrl = 0, 0  # 지정할 허리->발목 길이
    theta14_ctrl, theta24_ctrl = 0, 0  # 지정할 허리->발목 각도

    kick_dis = 0.01  # kick 할 정도 결정
    z_foot_kickStop = 0.01  # 킥 모션을 중지할 발 높이

    retract_dis = 0.01

    # get position and vel of joints
    theta0 = (data.qpos[right_pelvis_joint]+data.qpos[left_pelvis_joint])/2
    theta11 = data.qpos[right_pelvis_joint]
    theta21 = data.qpos[left_pelvis_joint]
    theta12 = data.qpos[right_knee_joint]
    theta22 = data.qpos[left_knee_joint]

    # State Estimation
    joint_no1 = right_pelvis_joint
    joint_no2 = right_foot_joint_1

    # l_14 계산
    l_14 = get_len_4(data.xanchor[joint_no1, 0], data.xanchor[joint_no1, 1], data.xanchor[joint_no1, 2], data.xanchor[joint_no2, 0], data.xanchor[joint_no2, 1], data.xanchor[joint_no2, 2])
    
    joint_no1 = left_pelvis_joint
    joint_no2 = left_foot_joint_1
    l_24 = get_len_4(data.xanchor[joint_no1, 0], data.xanchor[joint_no1, 1], data.xanchor[joint_no1, 2], data.xanchor[joint_no2, 0], data.xanchor[joint_no2, 1], data.xanchor[joint_no2, 2])

    abs_theta_leg1 = get_theta_n4(l_1, l_2, theta11, theta12)

    abs_theta_leg2 = get_theta_n4(l_1, l_2, theta21, theta22)

    #position of foot1
    body_no = foot1_body 
    z_foot1 = data.xpos[body_no, 2]

    body_no = foot2_body
    z_foot2 = data.xpos[body_no, 2]

    # Transition check
    if True:  # 그냥 코드 접으려고 if문 추가
        if fsm_hip == fsm_leg2_swing and z_foot2 < 0.05 and data.xanchor[right_foot_joint_1, 0]<data.xanchor[right_pelvis_joint, 0]:
            fsm_hip = fsm_leg1_swing

        if fsm_hip == fsm_leg1_swing and z_foot1 < 0.05 and data.xanchor[left_foot_joint_1, 0]<data.xanchor[left_pelvis_joint, 0]:
            fsm_hip = fsm_leg2_swing

        if fsm_knee1 == fsm_knee1_stance and z_foot2 < 0.05 and data.xanchor[right_foot_joint_1, 0]<data.xanchor[right_pelvis_joint, 0]:  # kick state for leg1
            fsm_knee1 = fsm_knee1_kick

        if fsm_knee1 == fsm_knee1_kick and z_foot1 > z_foot_kickStop and data.xanchor[right_foot_joint_1, 0]<data.xanchor[right_pelvis_joint, 0]:  # modified retract state for leg1
            fsm_knee1 = fsm_knee1_retract

        if fsm_knee1 == fsm_knee1_retract and abs_theta_leg1 > 0.1:
            fsm_knee1 = fsm_knee1_stance

        if fsm_knee2 == fsm_knee2_stance and z_foot1 < 0.05 and data.xanchor[left_foot_joint_1, 0]<data.xanchor[left_pelvis_joint, 0]:  # kick state for leg2
            fsm_knee2 = fsm_knee2_kick

        if fsm_knee2 == fsm_knee2_kick and z_foot2 > z_foot_kickStop and data.xanchor[left_foot_joint_1, 0]<data.xanchor[left_pelvis_joint, 0]:  # modified retract state for leg2
            fsm_knee2 = fsm_knee2_retract

        if fsm_knee2 == fsm_knee2_retract and abs_theta_leg2 > 0.1:
                fsm_knee2 = fsm_knee2_stance
                
    # All stabilizer here
    if True:
        # 여기다가 발바닥 각도 넣어주면 될듯.
        data.ctrl[anckle1_pservo_y] = -(theta0 + theta11 + theta12)
        data.ctrl[anckle2_pservo_y] = -(theta0 + theta21 + theta22)

        # All actions here
        if fsm_hip == fsm_leg1_swing:
            theta14_ctrl = 30*np.pi/180
            theta24_ctrl = -15*np.pi/180

        if fsm_hip == fsm_leg2_swing:
            theta14_ctrl = -15*np.pi/180
            theta24_ctrl = 30*np.pi/180

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


def init_controller(model,data):
    global theta0

    left=0
    if left==True:
        init_l1 = 0.1
        init_l2 = 0.1
        r_tmp_theta1, r_tmp_theta2 = get_leg_ctrl_radian(1,1,init_l1,0)
        data.ctrl[hip1_pservo_y] = r_tmp_theta1
        data.ctrl[knee1_pservo_y] = r_tmp_theta2
        data.ctrl[anckle1_pservo_y] = 0

        tmp_theta1, tmp_theta2 = get_leg_ctrl_radian(1,1,init_l2, 0.5)
        data.ctrl[hip2_pservo_y] = tmp_theta1
        data.ctrl[knee2_pservo_y] = tmp_theta2 #왼발 앞으로
        data.ctrl[anckle2_pservo_y] = -(theta0+tmp_theta1+tmp_theta2) #왼발바닥 바닥보게

        fsm_hip = fsm_leg2_swing
        fsm_knee1 = fsm_knee1_stance
        fsm_knee2 = fsm_knee2_stance
    
    else:
        init_l1 = 0.1
        init_l2 = 0.1
        r_tmp_theta1, r_tmp_theta2 = get_leg_ctrl_radian(1,1,init_l1,0.5)
        data.ctrl[hip1_pservo_y] = r_tmp_theta1
        data.ctrl[knee1_pservo_y] = r_tmp_theta2 #오른발 앞으로
        data.ctrl[anckle1_pservo_y] =  -(theta0+r_tmp_theta1+r_tmp_theta2)

        tmp_theta1, tmp_theta2 = get_leg_ctrl_radian(1,1,init_l2, 0)
        data.ctrl[hip2_pservo_y] = tmp_theta1
        data.ctrl[knee2_pservo_y] = tmp_theta2 
        data.ctrl[anckle2_pservo_y] = 0
        fsm_hip = fsm_leg1_swing
        fsm_knee1 = fsm_knee1_stance
        fsm_knee2 = fsm_knee2_stance

def calculateLength(first, second):
    x_diff = first[0] - second[0]
    y_diff = first[1] - second[1]
    z_diff = first[2] - second[2]
    return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2)+math.pow(z_diff,2))

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

def get_leg_ctrl_radian(l_1, l_2, l_ctrl, theta_input):
    theta_2 =  -math.acos((math.pow(l_ctrl,2)-math.pow(l_1,2)-math.pow(l_2,2))/(2*l_1*l_2))
    theta_1 = theta_input+math.acos((math.pow(l_1,2)+math.pow(l_ctrl,2)-math.pow(l_2,2))/(2*l_1*l_ctrl));
    
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
cam.elevation = -30  # -11.588379
cam.distance = 2.0  # 5.0
cam.lookat = np.array([0.0, 0.0, 0.2])

#turn the direction of gravity to simulate a ramp
gamma=0
gravity=9.81
model.opt.gravity[0] = gravity * np.sin(gamma) # downhill과 같이 중력 설정
model.opt.gravity[2] = -gravity * np.cos(gamma)

init_controller(model,data)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        #simulation step
        mj.mj_step(model, data)
        # Apply control
        controller(model, data)

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
