

#include<stdbool.h> //for bool
//#include<unistd.h> //for usleep
//#include <math.h>

#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "myUtility.c"

int fsm_hip;
int fsm_knee1;
int fsm_knee2;

#define fsm_leg1_swing 0
#define fsm_leg2_swing 1

#define fsm_knee1_stance 0
#define fsm_knee1_retract 1
#define fsm_knee1_kick 2

#define fsm_knee2_stance 0
#define fsm_knee2_retract 1
#define fsm_knee2_kick 2

// 자꾸 다른 파일 열어보기 귀찮아서 키워드설정 for joints
#define x_joint 0
#define z_joint 1
#define pin_joint_y 2
#define hip1_joint_y 3
#define knee1_joint_y 4
#define anckle1_joint_y 5
#define hip2_joint_y 6
#define knee2_joint_y 7
#define anckle2_joint_y 8

// 키워드설정 for actuators
#define hip1_pservo_y 0
#define hip1_vservo_y 1
#define hip2_pservo_y 2
#define hip2_vservo_y 3
#define knee1_pservo_y 4
#define knee1_vservo_y 5
#define knee2_pservo_y 6
#define knee2_vservo_y 7
#define anckle1_pservo_y 8
#define anckle1_vservo_y 9
#define anckle2_pservo_y 10
#define anckle2_vservo_y 11

// 키워드설정 for bodies
#define world_body 0
#define ass_body 1
#define leg11_body 2
#define leg12_body 3
#define foot1_body 4
#define leg21_body 5
#define leg22_body 6
#define foot2_body 7

//simulation end time
double simend = 20;

//related to writing data to a file
FILE *fid;
int loop_index = 0;
const int data_frequency = 10; //frequency at which data is written to a file


// char xmlpath[] = "../myproject/template_writeData/pendulum.xml";
// char datapath[] = "../myproject/template_writeData/data.csv";


//Change the path <template_writeData>
//Change the xml file
char path[] = "../myproject/biped/";
char xmlfile[] = "biped.xml";


char datafile[] = "data.csv";


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


//****************************
//This function is called once and is used to get the headers
void init_save_data()
{
  //write name of the variable here (header)
   fprintf(fid,"t, ");

   //Don't remove the newline
   fprintf(fid,"\n");
}

//***************************
//This function is called at a set frequency, put data here
void save_data(const mjModel* m, mjData* d)
{
  //data here should correspond to headers in init_save_data()
  //seperate data by a space %f followed by space
  fprintf(fid,"%f ",d->time);

  //Don't remove the newline
  fprintf(fid,"\n");
}

/******************************/
void set_torque_control(const mjModel* m,int actuator_no,int flag)
{
  if (flag==0)
    m->actuator_gainprm[10*actuator_no+0]=0;
  else
    m->actuator_gainprm[10*actuator_no+0]=1;
}
/******************************/


/******************************/
void set_position_servo(const mjModel* m,int actuator_no,double kp)
{
  m->actuator_gainprm[10*actuator_no+0]=kp;
  m->actuator_biasprm[10*actuator_no+1]=-kp;
}
/******************************/

/******************************/
void set_velocity_servo(const mjModel* m,int actuator_no,double kv)
{
  m->actuator_gainprm[10*actuator_no+0]=kv;
  m->actuator_biasprm[10*actuator_no+2]=-kv;
}
/******************************/

//**************************
void init_controller(const mjModel* m, mjData* d)
{
    // leg2 한쪽 발 앞으로 뻗는 자세 만들어줘야 함. 

    //d->qpos[hip2_joint_y] = 0.5;
    //d->ctrl[hip2_joint_y] = d->qpos[hip2_joint_y]; // 왼발 앞으로 하고 시작

    double init_l1 = 1.75;
    double init_l2 = 1.75;
    double tmp_theta1, tmp_theta2;
    getLegCtrlRadian(1,1,init_l1,0,&tmp_theta1,&tmp_theta2);
    d->ctrl[hip1_pservo_y] = tmp_theta1;
    d->ctrl[knee1_pservo_y] = tmp_theta2;
    d->ctrl[anckle1_pservo_y] = 0;

    getLegCtrlRadian(1,1,init_l2, 0.5, &tmp_theta1, &tmp_theta2);
    d->ctrl[hip2_pservo_y] = tmp_theta1;
    d->ctrl[knee2_pservo_y] = tmp_theta2; //왼발 앞으로
    d->ctrl[anckle2_pservo_y] = -(qpos[pin_joint_y]+tmp_theta1+tmp_theta2); //왼발바닥 바닥보게

  fsm_hip = fsm_leg2_swing;
  fsm_knee1 = fsm_knee1_stance;
  fsm_knee2 = fsm_knee2_stance;
  
}

//**************************
void mycontroller(const mjModel* m, mjData* d)
{
  //instant variable
  int body_no;
  int joint_no1, joint_no2;
  double pos_x1, pos_y1, pos_z1;
  double pos_x2, pos_y2, pos_z2;
  double quat0, quatx, quaty, quatz;
  double euler_x1, euler_y1, euler_z1;
  double euler_x2, euler_y2, euler_z2;

  // 변수 정의
  double l_1 =1; double l_2=1;double l_3=0.1; //다리 길이
  double l_stance = 1.75; //서있을 때 펴고 있을 다리 길이
  double theta14, theta14dot; //허리->발목 벡터와 몸통 z축 사잇각 for leg1
  double theta24, theta24dot; 
  double abs_theta_leg1, abs_theta_leg2; // world 좌표계에서 다리 각도
  double l_14, l_24; //허리->발목 길이
  double z_foot1, z_foot2;

  double theta11_ctrl, theta12_ctrl, theta13_ctrl;//지정할 다리 부분 조인트각도
  double theta21_ctrl, theta22_ctrl, theta23_ctrl;
  double l_14_ctrl, l_24_ctrl; //지정할 허리->발목 길이
  double theta14_ctrl, theta24_ctrl; //지정할 허리->발목 각도
  

  double kick_dis = 0.1; //kick 할 정도 결정
  double z_foot_kickStop = 0.05; //킥 모션을 중지할 발 높이

  double retract_dis = 0.3;

  //get position and vel of joints
  double x = d->qpos[x_joint];double vx = d->qvel[x_joint];
  double z = d->qpos[z_joint]; double vz = d->qvel[z_joint];
  double theta0 = d->qpos[pin_joint_y]; double theta0dot = d->qvel[pin_joint_y];
  double theta11 = d->qpos[hip1_joint_y]; double theta11dot = d->qvel[hip1_joint_y];
  double theta21 = d->qpos[hip2_joint_y]; double theta21dot = d->qvel[hip2_joint_y];
  double theta12 = d->qpos[knee1_joint_y]; double theta12dot = d->qvel[knee1_joint_y];
  double theta22 = d->qpos[knee2_joint_y]; double theta22dot = d->qvel[knee2_joint_y];
  double theta13 = d->qpos[anckle1_joint_y]; double theta13dot = d->qvel[anckle1_joint_y];
  double theta23 = d->qpos[anckle2_joint_y]; double theta23dot = d->qvel[anckle2_joint_y];
  
  //state estimation(bodies)
  //각 다리에 대해 l_@4와 theta_@4를 계산
  joint_no1 = hip1_joint_y;
  joint_no2 = anckle1_joint_y;
  getLn4(d->xanchor[3*joint_no1],d->xanchor[3*joint_no1+1],d->xanchor[3*joint_no1+2],
         d->xanchor[3*joint_no2],d->xanchor[3*joint_no2+1],d->xanchor[3*joint_no2+2], &l_14); //l_14 계산
  
  joint_no1 = hip2_joint_y;
  joint_no2 = anckle2_joint_y;  
  getLn4(d->xanchor[3*joint_no1],d->xanchor[3*joint_no1+1],d->xanchor[3*joint_no1+2],
         d->xanchor[3*joint_no2],d->xanchor[3*joint_no2+1],d->xanchor[3*joint_no2+2], &l_24); //l_14 계산
  
  getThetan4(l_1, l_2, theta11, theta12, &theta14);
  abs_theta_leg1 = theta0+theta14;

  getThetan4(l_1, l_2, theta21, theta22, &theta24);
  abs_theta_leg2 = theta0+theta24;  

  //position of foot1
  body_no = foot1_body;
  //x = d->xpos[3*body_no]; y = d->qpos[3*body_no+1]; 
  z_foot1 = d->xpos[3*body_no+2];
  //printf("%f \n", z_foot1);

  body_no = foot2_body;
  z_foot2 = d->xpos[3*body_no+2];

  //All transitions here
  if(true){ //그냥 코드 접으려고 if문 추가
  if(fsm_hip == fsm_leg2_swing && z_foot2<0.05 && abs_theta_leg1 <0)
  {
    fsm_hip = fsm_leg1_swing;
  }
  if(fsm_hip == fsm_leg1_swing && z_foot1<0.05 && abs_theta_leg2<0)
  {
    fsm_hip = fsm_leg2_swing;
  }

  if(fsm_knee1 == fsm_knee1_stance && z_foot2 <0.05 && abs_theta_leg1<0) // kick state for leg1
  {
    fsm_knee1 = fsm_knee1_kick;
  }
  if (fsm_knee1 == fsm_knee1_kick && z_foot1 > z_foot_kickStop && abs_theta_leg1<0) // modified retract state for leg1
  {
    fsm_knee1 = fsm_knee1_retract;
  }
  if(fsm_knee1 == fsm_knee1_retract && abs_theta_leg1>0.1)
  {
    fsm_knee1 = fsm_knee1_stance;
  }

    if(fsm_knee2 == fsm_knee2_stance && z_foot1 <0.05 && abs_theta_leg2<0) // kick state for leg2
  {
    fsm_knee2 = fsm_knee2_kick;
  }
  if (fsm_knee2 == fsm_knee2_kick && z_foot2 > z_foot_kickStop && abs_theta_leg2<0) // modified retract state for leg2
  {
    fsm_knee2 = fsm_knee2_retract;
  }
  if(fsm_knee2 == fsm_knee2_retract && abs_theta_leg2>0.1)
  {
    fsm_knee2 = fsm_knee2_stance;
  }
  }

  // All stabilizer here
  if(true){
    // 여기다가 발바닥 각도 넣어주면 될듯. 
  }

  //All actions here
  if (fsm_hip == fsm_leg1_swing)
  {
    theta14_ctrl = -0.25;
    theta24_ctrl =  0.25;
    
    //d->ctrl[0] = -0.5; //xml에 있는 actuator no에 값 지정
  }
  if (fsm_hip == fsm_leg2_swing)
  {
    theta14_ctrl = 0.25;
    theta24_ctrl = -0.25;
    //d->ctrl[0] = 0.5;
  }

  if (fsm_knee1 == fsm_knee1_stance)
  {
    l_14_ctrl = l_stance;

    //d->ctrl[2] = 0;
  }
  if (fsm_knee1 == fsm_knee1_kick) //state가 kick일때 살짝 발차기
  {
    l_14_ctrl = l_stance+kick_dis;

    //d->ctrl[2] = kick_dis;
  }
  if (fsm_knee1 == fsm_knee1_retract)
  {
    l_14_ctrl = l_stance - retract_dis;
    
    //d->ctrl[2] = -0.25;
  }

  if (fsm_knee2 == fsm_knee2_stance)
  {
    l_24_ctrl = l_stance;

    //d->ctrl[4] = 0;
  }
  if (fsm_knee2 == fsm_knee2_kick)
  {
    l_24_ctrl = l_stance+kick_dis;
    
    //d->ctrl[4] = kick_dis;
  }
  if (fsm_knee2 == fsm_knee2_retract)
  {
    l_24_ctrl = l_stance - retract_dis;
    
    //d->ctrl[4] = -0.25;
  }
   //action for leg 1
   getLegCtrlRadian(l_1, l_2, l_14_ctrl, theta14_ctrl,&theta11_ctrl, &theta12_ctrl);
   d->ctrl[hip1_pservo_y] = theta11_ctrl;
   d->ctrl[knee1_pservo_y] = theta12_ctrl;

   //action for leg 2
   getLegCtrlRadian(l_1, l_2, l_24_ctrl, theta24_ctrl, &theta11_ctrl, &theta12_ctrl);
   d->ctrl[hip2_pservo_y] = theta11_ctrl;
   d->ctrl[knee2_pservo_y] = theta12_ctrl;

  //write data here (dont change/dete this function call; instead write what you need to save in save_data)
  if ( loop_index%data_frequency==0)
    {
      save_data(m,d);
    }
  loop_index = loop_index + 1;
}


//************************
// main function
int main(int argc, const char** argv)
{

    // activate software
    mj_activate("mjkey.txt");

    char xmlpath[100]={};
    char datapath[100]={};

    strcat(xmlpath,path);
    strcat(xmlpath,xmlfile);

    strcat(datapath,path);
    strcat(datapath,datafile);


    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
        m = mj_loadXML(xmlpath, 0, error, 1000);

    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);


    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    //double arr_view[] = {89.608063, -11.588379, 8, 0.000000, 0.000000, 2.000000}; //view the left side (for ll, lh, left_side)
    double arr_view[] = {140.779492, -28.302665, 9.180318, 1.650492, 1.787461, -0.298866};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // install control callback
    mjcb_control = mycontroller;

    fid = fopen(datapath,"w");
    init_save_data();
    init_controller(m,d);

    //내리막길의 중력을 없애보자

    double gamma = 0; //set ramp slope 0
    double gravity = 9.81;
    m->opt.gravity[2] = -gravity*cos(gamma);
    m->opt.gravity[0] = gravity*sin(gamma);

    // use the first while condition if you want to simulate for a period.
    while( !glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            mj_step(m, d);
        }

        if (d->time>=simend)
        {
           fclose(fid);
           break;
         }

       // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        //opt.frame = mjFRAME_WORLD; //mjFRAME_BODY
        //opt.flags[mjVIS_COM]  = 1 ; //mjVIS_JOINT;
        opt.flags[mjVIS_JOINT]  = 1 ;
          // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        //int body = 2;
        //cam.lookat[0] = d->xpos[3*body+0];
        //printf("{%f, %f, %f, %f, %f, %f};\n",cam.azimuth,cam.elevation, cam.distance,cam.lookat[0],cam.lookat[1],cam.lookat[2]);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
