/* Null distribution of the Hurst estimator at their series length (n=1000/5000).
   If i.i.d. (true H=0) or random walk (H=0.5) already produce a >0.6 tail,
   then Living's "18% human runs" is unremarkable. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double gauss(void){double u=0,v=0;while(!u)u=(rand()+1.0)/(RAND_MAX+2.0);
    while(!v)v=(rand()+1.0)/(RAND_MAX+2.0);return sqrt(-2*log(u))*cos(2*M_PI*v);}
static double hurst(const double*x,int n){
    int hi=(n/2<50?n/2:50);double lx[64],ly[64];int m=0;
    for(int lag=2;lag<hi;lag++){double a=0;for(int i=lag;i<n;i++){double d=x[i]-x[i-lag];a+=d*d;}
        double t=sqrt(a/(n-lag)); if(t>1e-12){lx[m]=log((double)lag);ly[m]=log(t);m++;}}
    if(m<2)return 0.5;double sx=0,sy=0,sxx=0,sxy=0;
    for(int i=0;i<m;i++){sx+=lx[i];sy+=ly[i];sxx+=lx[i]*lx[i];sxy+=lx[i]*ly[i];}
    return (m*sxy-sx*sy)/(m*sxx-sx*sx);}
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):1000, R=argc>2?atoi(argv[2]):2000;
    srand(7);
    for(int kind=0; kind<3; kind++){
        double sum=0,sum2=0,mx=-9;int g6=0,g5=0;
        double *x=malloc(sizeof(double)*n);
        for(int r=0;r<R;r++){
            if(kind==0){ for(int i=0;i<n;i++) x[i]=gauss(); }                 /* i.i.d. H=0   */
            else if(kind==1){ double s=0; for(int i=0;i<n;i++){s+=gauss();x[i]=s;} } /* RW H=.5 */
            else { x[0]=gauss(); for(int i=1;i<n;i++) x[i]=0.9*x[i-1]+gauss(); }     /* AR(1,.9) */
            double h=hurst(x,n); sum+=h;sum2+=h*h; if(h>0.6)g6++; if(h>0.5)g5++; if(h>mx)mx=h;
        }
        double m=sum/R,var=sum2/R-m*m;
        const char*name = kind==0?"i.i.d.(H_true=0)":kind==1?"random-walk(H=.5)":"AR(1,.9)";
        printf("n=%5d %-20s H=%.3f±%.3f  >0.5:%3.0f%%  >0.6:%3.0f%%  max=%.3f\n",
               n,name,m,sqrt(var<0?0:var),100.0*g5/R,100.0*g6/R,mx);
        free(x);
    }
    return 0;
}
