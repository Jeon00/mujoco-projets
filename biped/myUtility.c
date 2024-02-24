#include <math.h>

//direction cos to euler angle
void mat2euler(double dircos[3][3],
    double *a1,
    double *a2,
    double *a3) 
{
    double th1,th2,th3,temp[10];

    if (((fabs(dircos[0][2])-1.) >= -1e-15)  ) {
        th1 = atan2(dircos[2][1],dircos[1][1]);
        if ((dircos[0][2] > 0.)  ) {
            temp[0] = 1.5707963267949;
        } else {
            temp[0] = -1.5707963267949;
        }
        th2 = temp[0];
        th3 = 0.;
    } else {
        th1 = atan2(-dircos[1][2],dircos[2][2]);
        th2 = asin(dircos[0][2]);

        th3 = atan2(-dircos[0][1],dircos[0][0]);
    }
    *a1 = th1;
    *a2 = th2;
    *a3 = th3;
}

//direction cos to uat angle
void mat2quat(double dircos[3][3],
     double *e4, //const
     double *e1, //qx
     double *e2, //qy
     double *e3) //qz
{
    double tmp,tmp1,tmp2,tmp3,tmp4,temp[10];

    tmp = (dircos[0][0]+(dircos[1][1]+dircos[2][2]));
    if (((tmp >= dircos[0][0]) && ((tmp >= dircos[1][1]) && (tmp >= dircos[2][2]
      )))  ) {
        tmp1 = (dircos[2][1]-dircos[1][2]);
        tmp2 = (dircos[0][2]-dircos[2][0]);
        tmp3 = (dircos[1][0]-dircos[0][1]);
        tmp4 = (1.+tmp);
    } else {
        if (((dircos[0][0] >= dircos[1][1]) && (dircos[0][0] >= dircos[2][2]))
          ) {
            tmp1 = (1.-(tmp-(2.*dircos[0][0])));
            tmp2 = (dircos[0][1]+dircos[1][0]);
            tmp3 = (dircos[0][2]+dircos[2][0]);
            tmp4 = (dircos[2][1]-dircos[1][2]);
        } else {
            if ((dircos[1][1] >= dircos[2][2])  ) {
                tmp1 = (dircos[0][1]+dircos[1][0]);
                tmp2 = (1.-(tmp-(2.*dircos[1][1])));
                tmp3 = (dircos[1][2]+dircos[2][1]);
                tmp4 = (dircos[0][2]-dircos[2][0]);
            } else {
                tmp1 = (dircos[0][2]+dircos[2][0]);
                tmp2 = (dircos[1][2]+dircos[2][1]);
                tmp3 = (1.-(tmp-(2.*dircos[2][2])));
                tmp4 = (dircos[1][0]-dircos[0][1]);
            }
        }
    }
    tmp = (1./sqrt(((tmp1*tmp1)+((tmp2*tmp2)+((tmp3*tmp3)+(tmp4*tmp4))))));
    *e1 = (tmp*tmp1);
    *e2 = (tmp*tmp2);
    *e3 = (tmp*tmp3);
    *e4 = (tmp*tmp4);
}

void euler2mat(double a1,
    double a2,
    double a3,
    double dircos[3][3])
{
    double cos1,cos2,cos3,sin1,sin2,sin3;

    cos1 = cos(a1);
    cos2 = cos(a2);
    cos3 = cos(a3);
    sin1 = sin(a1);
    sin2 = sin(a2);
    sin3 = sin(a3);
    dircos[0][0] = (cos2*cos3);
    dircos[0][1] = -(cos2*sin3);
    dircos[0][2] = sin2;
    dircos[1][0] = ((cos1*sin3)+(sin1*(cos3*sin2)));
    dircos[1][1] = ((cos1*cos3)-(sin1*(sin2*sin3)));
    dircos[1][2] = -(cos2*sin1);
    dircos[2][0] = ((sin1*sin3)-(cos1*(cos3*sin2)));
    dircos[2][1] = ((cos1*(sin2*sin3))+(cos3*sin1));
    dircos[2][2] = (cos1*cos2);
}

void quat2mat(double ie4, //constant
    double ie1, //qx
    double ie2, //qy
    double ie3, //qz
    double dircos[3][3])
{
    double e1,e2,e3,e4,e11,e22,e33,e44,norm;

    e11 = ie1*ie1;
    e22 = ie2*ie2;
    e33 = ie3*ie3;
    e44 = ie4*ie4;
    norm = sqrt(e11+e22+e33+e44);
    if (norm == 0.) {
        e4 = 1.;
        norm = 1.;
    } else {
        e4 = ie4;
    }
    norm = 1./norm;
    e1 = ie1*norm;
    e2 = ie2*norm;
    e3 = ie3*norm;
    e4 = e4*norm;
    e11 = e1*e1;
    e22 = e2*e2;
    e33 = e3*e3;
    dircos[0][0] = 1.-(2.*(e22+e33));
    dircos[0][1] = 2.*(e1*e2-e3*e4);
    dircos[0][2] = 2.*(e1*e3+e2*e4);
    dircos[1][0] = 2.*(e1*e2+e3*e4);
    dircos[1][1] = 1.-(2.*(e11+e33));
    dircos[1][2] = 2.*(e2*e3-e1*e4);
    dircos[2][0] = 2.*(e1*e3-e2*e4);
    dircos[2][1] = 2.*(e2*e3+e1*e4);
    dircos[2][2] = 1.-(2.*(e11+e22));
}

void euler2quat(double a1, double a2, double a3, double *e4, double *e1, double *e2, double *e3)
{
double dircos[3][3];
double tmp1,tmp2,tmp3,tmp4;
euler2mat(a1,a2,a3,dircos);
mat2quat(dircos,&tmp4,&tmp1,&tmp2,&tmp3);
*e4 = tmp4; //constant
*e1 = tmp1;
*e2 = tmp2;
*e3 = tmp3;
}

void quat2euler(double e4, double e1, double e2, double e3, double *a1, double *a2, double *a3)
{
double dircos[3][3];
double tmp1,tmp2,tmp3;
quat2mat(e4,e1,e2,e3,dircos);
mat2euler(dircos,&tmp1,&tmp2,&tmp3);
*a1 = tmp1;
*a2 = tmp2;
*a3 = tmp3;
}
void getLn4(double x1,double y1,double z1,double x2,double y2,double z2,double *len){
    double tmp1,tmp2,tmp3;
    tmp1 = x2-x1;
    tmp2 = y2-y1;
    tmp3 = z2-z1;
    *len = sqrt(tmp1^2 + tmp2^2 + tmp3^2);
}

void getThetan4(double l1, double l2, double theta1, double theta2, double *resultTheta){
    *resultTheta = atan2(l1*sin(theta1)-l2*sin(theta2-theta1) , l1*cos(theta1)+l2*cos(theta2-theta1));
}

void getAnckleCtrlRadian(){// 스테빌라이저에 넣을 함수

}

void getLegCtrlRadian(double l_1, double l_2, double l_ctrl, double theta_input, double *theta_output1, double *theta_output2){
    // Link lengths
    double a = l_1;
    double b = l_2;
    
    // End effector position
    double x = l_ctrl * cos(theta_input);
    double y = l_ctrl * sin(theta_input);

    // Calculate theta_2
    double D = (x * x + y * y - a * a - b * b) / (2 * a * b);
    double theta_2 = atan2(-sqrt(1 - D * D), D);

    // Calculate theta_1
    double theta_1 = atan2(y, x) - atan2(b * sin(theta_2), a + b * cos(theta_2));

    // Return results
    *theta_output1 = theta_1;
    *theta_output2 = theta_2;

}

