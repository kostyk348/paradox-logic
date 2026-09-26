/* Fast C port of Living + Hurst (Untitled278 LivingSystem@10308).
   5 base ops, meta-rule evolution by COMPOSITION C(s)=tanh(.5*P(s)+.5*P(P(s))).
   Cost of a composed rule = 2^depth -> unbounded original is intractable
   (notebook v38 aborted). Depth capped at MAXD.
   Hurst = slope log(RMS increment) vs log(lag), identical to their estimator.
   gcc -O2 -o living living.c -lm                                               */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define K 4          /* state_dim */
#define MAXD 8
#define MAXRULE 8192

typedef struct { int op; int parent; int depth; } Rule;
static Rule rules[MAXRULE];
static int n_rules;

static void apply_op(int op, const double *s, const double *inp, double *out){
    for(int d=0; d<K; d++){
        double sl = s[(d-1+K)%K];          /* np.roll(s,1)[d] = s[d-1] */
        switch(op){
            case 0: out[d]=tanh(s[d]+0.1*inp[d]); break;
            case 1: out[d]=tanh(0.9*s[d]-0.2*inp[d]); break;
            case 2: out[d]=sin(s[d]+inp[d]); break;
            case 3: out[d]=tanh(s[d]*inp[d]); break;
            case 4: out[d]=tanh(s[d]+0.3*sl); break;
        }
    }
}
static void eval_rule(int id, const double *s, const double *inp, double *out){
    Rule *r=&rules[id];
    if(r->op<=4){ apply_op(r->op,s,inp,out); return; }
    double a[K], b[K];
    eval_rule(r->parent, s,   inp, a);
    eval_rule(r->parent, a,   inp, b);
    for(int d=0; d<K; d++) out[d]=tanh(0.5*a[d]+0.5*b[d]);
}
static double gauss(void){ double u=0,v=0; while(u==0)u=(rand()+1.0)/(RAND_MAX+2.0);
    while(v==0)v=(rand()+1.0)/(RAND_MAX+2.0); return sqrt(-2*log(u))*cos(2*M_PI*v); }

static double hurst(const double *x,int n){
    int hi=(n/2<50?n/2:50); double lx[64],ly[64]; int m=0;
    for(int lag=2; lag<hi; lag++){
        double acc=0; int c=0;
        for(int i=lag;i<n;i++){ double d=x[i]-x[i-lag]; acc+=d*d; c++; }
        double tau=sqrt(acc/c);
        if(tau>1e-12){ lx[m]=log((double)lag); ly[m]=log(tau); m++; }
    }
    if(m<2) return 0.5;
    double sx=0,sy=0,sxx=0,sxy=0;
    for(int i=0;i<m;i++){sx+=lx[i];sy+=ly[i];sxx+=lx[i]*lx[i];sxy+=lx[i]*ly[i];}
    return (m*sxy-sx*sy)/(m*sxx-sx*sx);
}
static void init_rules(void){ n_rules=0; for(int o=0;o<5;o++) rules[n_rules++]=(Rule){o,-1,0}; }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):8;
    double meta=argc>2?atof(argv[2]):0.05;
    int STEPS=argc>3?atoi(argv[3]):1000;
    int RUNS=argc>4?atoi(argv[4]):500;
    int allcomp=argc>5?atoi(argv[5]):0;
    srand(2026);
    double sum=0,sum2=0,mx=-9; int g6=0,g5=0;
    static double inp[64][K], ns[64][K];
    for(int run=0; run<RUNS; run++){
        init_rules();
        static double st[64][K]; static int rid[64];
        for(int i=0;i<N;i++){ for(int d=0;d<K;d++) st[i][d]=gauss(); rid[i]=rand()%5; }
        double *ser=malloc(sizeof(double)*STEPS);
        double *acc=allcomp?calloc(STEPS,sizeof(double)):NULL;
        for(int t=0;t<STEPS;t++){
            for(int i=0;i<N;i++) for(int d=0;d<K;d++){
                double s=0; for(int j=0;j<N;j++) s+=st[j][d];
                inp[i][d]=(s-st[i][d])/(N-1);
            }
            for(int i=0;i<N;i++) eval_rule(rid[i], st[i], inp[i], ns[i]);
            memcpy(st,ns,sizeof(double)*N*K);
            if(meta>0){
                for(int i=0;i<N;i++){
                    if(((double)rand()/RAND_MAX)<meta){
                        int id=n_rules++;
                        if(n_rules>=MAXRULE) n_rules=5;
                        rules[id]=(Rule){5,rid[i],rules[rid[i]].depth+1};
                        if(rules[id].depth>MAXD){ rules[id]=(Rule){rand()%5,-1,0}; }
                        rid[i]=id;
                    }
                }
            }
            ser[t]=st[0][0];
            if(allcomp){ for(int i=0;i<N;i++) acc[t]+=st[i][0]; }
        }
        double h;
        if(allcomp){ for(int t=0;t<STEPS;t++) acc[t]/=N; h=hurst(acc,STEPS);} else h=hurst(ser,STEPS);
        sum+=h; sum2+=h*h; if(h>0.6)g6++; if(h>0.5)g5++; if(h>mx)mx=h;
        free(ser); if(acc)free(acc);
    }
    double mean=sum/RUNS, var=sum2/RUNS-mean*mean;
    printf("N=%2d meta=%.2f steps=%5d runs=%4d %s : H=%.3f±%.3f  >0.5:%3.0f%%  >0.6:%3.0f%%  max=%.3f\n",
        N,meta,STEPS,RUNS, allcomp?"allcomp":"comp0 ",mean,sqrt(var<0?0:var),
        100.0*g5/RUNS,100.0*g6/RUNS,mx);
    return 0;
}
