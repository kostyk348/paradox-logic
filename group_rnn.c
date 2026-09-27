/* group_rnn.c — group-valued recurrent net vs MLP, in C (no framework).
 *
 * GroupRNN: recurrent state = probability distribution over a finite group Γ, composed
 * through the regular representation of right multiplication.  It exactly implements the
 * holonomy of the token sequence.  Hypothesis (algorithmic Whorf): it wins on tasks whose
 * ground-truth algebra IS the group (parity=Z/2, sum mod k=Z/k, permutations=S3) and gives
 * no advantage on MNIST, whose algebra is not a group.
 *
 *   gcc -O2 -o group_rnn group_rnn.c -lm
 *   ./group_rnn [path_to_mnist_dir]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ---------- rng ---------- */
static unsigned long long RS = 88172645463325252ULL;
static float urand(void){ RS ^= RS<<13; RS ^= RS>>7; RS ^= RS<<17;
    return (float)((RS>>11)&0xFFFFFF)/16777216.0f; }
static float nrand(void){ float u=urand(); if(u<1e-8f)u=1e-8f; float v=urand();
    return sqrtf(-2.f*logf(u))*cosf(6.2831853f*v); }

/* ---------- group ---------- */
typedef struct { int n; int mul[64][64]; int id; } Group;
static Group mk_cyclic(int k){ Group g; g.n=k; g.id=0;
    for(int a=0;a<k;a++)for(int b=0;b<k;b++) g.mul[a][b]=(a+b)%k; return g; }
static Group mk_sym3(void){
    int P[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    Group g; g.n=6; g.id=0;
    for(int a=0;a<6;a++)for(int b=0;b<6;b++){ int r[3]; for(int i=0;i<3;i++) r[i]=P[a][P[b][i]];
        for(int c=0;c<6;c++) if(P[c][0]==r[0]&&P[c][1]==r[1]&&P[c][2]==r[2]){ g.mul[a][b]=c; break; } }
    return g;
}
static void reg_rep(Group*g,float*P){ int n=g->n; memset(P,0,4*n*n*n);
    for(int gg=0;gg<n;gg++)for(int k=0;k<n;k++){ int h=g->mul[k][gg]; P[(gg*n+h)*n+k]=1.f; } }

/* ---------- params + adam ---------- */
typedef struct { int n; float *w,*gr,*m,*v; } P;
static long ADAM_T=0;
static void pinit(P*p,int n,float s){ p->n=n; p->w=malloc(4*n); p->gr=calloc(n,4); p->m=calloc(n,4); p->v=calloc(n,4);
    for(int i=0;i<n;i++) p->w[i]=nrand()*s; }
static void pzero(P*p){ memset(p->gr,0,4*p->n); }
static void padam(P*p,float lr){ for(int i=0;i<p->n;i++){
    p->m[i]=0.9f*p->m[i]+0.1f*p->gr[i]; p->v[i]=0.999f*p->v[i]+0.001f*p->gr[i]*p->gr[i];
    float mh=p->m[i]/(1.f-powf(0.9f,(float)ADAM_T)), vh=p->v[i]/(1.f-powf(0.999f,(float)ADAM_T));
    p->w[i]-=lr*mh/(sqrtf(vh)+1e-8f); } }
static void softmax(const float*z,float*q,int n){ float mx=z[0]; for(int i=1;i<n;i++) if(z[i]>mx)mx=z[i];
    float s=0; for(int i=0;i<n;i++){ q[i]=expf(z[i]-mx); s+=q[i]; } for(int i=0;i<n;i++) q[i]/=s; }

#define MAXT 160
/* ================= GroupRNN ================= */
typedef struct {
    Group G; int order,din,nout,T; float *Preg;
    P Wx,Wo,bo; float *x,*q,*pt,*dpt; float logits[32];
} GRNN;
static void grnn_init(GRNN*m,Group G,int din,int nout,float s){
    m->G=G; m->order=G.n; m->din=din; m->nout=nout;
    m->Preg=malloc(4*m->order*m->order*m->order); reg_rep(&G,m->Preg);
    pinit(&m->Wx,m->order*din,s); pinit(&m->Wo,nout*2*m->order,s); pinit(&m->bo,nout,0.01f);
    m->x=malloc(4*MAXT*din); m->q=malloc(4*MAXT*m->order);
    m->pt=malloc(4*(MAXT+1)*m->order); m->dpt=calloc((MAXT+1)*m->order,4);
}
static float grnn_forward(GRNN*m,int T,const float*x,int label){
    m->T=T; memcpy(m->x,x,4*T*m->din); int O=m->order;
    for(int i=0;i<O;i++) m->pt[i]=0; m->pt[m->G.id]=1.f;
    for(int t=0;t<T;t++){
        float z[64]; for(int o=0;o<O;o++){ float s=0; for(int j=0;j<m->din;j++) s+=m->Wx.w[o*m->din+j]*x[t*m->din+j]; z[o]=s; }
        float*qt=m->q+t*O; softmax(z,qt,O);
        float*pp=m->pt+t*O,*pn=m->pt+(t+1)*O;
        for(int h=0;h<O;h++){ float s=0; for(int gg=0;gg<O;gg++){ float qg=qt[gg];
            const float*Pg=m->Preg+(gg*O+h)*O; for(int k=0;k<O;k++) s+=qg*Pg[k]*pp[k]; } pn[h]=s; }
    }
    int F=2*O; float feat[64];
    for(int h=0;h<O;h++) feat[h]=m->pt[T*O+h];
    for(int h=0;h<O;h++){ float s=0; for(int t=1;t<=T;t++) s+=m->pt[t*O+h]; feat[O+h]=s/(float)T; }
    for(int o=0;o<m->nout;o++){ float s=m->bo.w[o]; for(int i=0;i<F;i++) s+=m->Wo.w[o*F+i]*feat[i]; m->logits[o]=s; }
    float pr[32]; softmax(m->logits,pr,m->nout);
    return -logf(pr[label]+1e-9f);
}
static void grnn_backward(GRNN*m,int label){
    int O=m->order,T=m->T,F=2*O; float feat[64];
    for(int h=0;h<O;h++) feat[h]=m->pt[T*O+h];
    for(int h=0;h<O;h++){ float s=0; for(int t=1;t<=T;t++) s+=m->pt[t*O+h]; feat[O+h]=s/(float)T; }
    float pr[32]; softmax(m->logits,pr,m->nout);
    float dl[32]; for(int o=0;o<m->nout;o++) dl[o]=pr[o]-(o==label?1.f:0.f);
    float dfeat[64]={0};
    for(int o=0;o<m->nout;o++){ m->bo.gr[o]+=dl[o];
        for(int i=0;i<F;i++){ m->Wo.gr[o*F+i]+=dl[o]*feat[i]; dfeat[i]+=m->Wo.w[o*F+i]*dl[o]; } }
    memset(m->dpt,0,4*(T+1)*O);
    for(int h=0;h<O;h++){ m->dpt[T*O+h]+=dfeat[h]; }
    for(int t=1;t<=T;t++) for(int h=0;h<O;h++) m->dpt[t*O+h]+=dfeat[O+h]/(float)T;
    for(int t=T;t>=1;t--){
        float*qt=m->q+(t-1)*O,*pp=m->pt+(t-1)*O,*dp=m->dpt+t*O;
        float dq[64]; for(int gg=0;gg<O;gg++){ float s=0;
            for(int h=0;h<O;h++){ float v=0; const float*Pgh=m->Preg+(gg*O+h)*O; for(int k=0;k<O;k++) v+=Pgh[k]*pp[k]; s+=dp[h]*v; } dq[gg]=s; }
        float dot=0; for(int gg=0;gg<O;gg++) dot+=dq[gg]*qt[gg];
        float dz[64]; for(int gg=0;gg<O;gg++) dz[gg]=qt[gg]*(dq[gg]-dot);
        for(int o=0;o<O;o++){ float d=dz[o]; if(d==0)continue; for(int j=0;j<m->din;j++) m->Wx.gr[o*m->din+j]+=d*m->x[(t-1)*m->din+j]; }
        float*dpm=m->dpt+(t-1)*O;
        for(int k=0;k<O;k++){ float acc=0; for(int h=0;h<O;h++){ float mhk=0;
            for(int gg=0;gg<O;gg++) mhk+=qt[gg]*m->Preg[(gg*O+h)*O+k]; acc+=dp[h]*mhk; } dpm[k]+=acc; }
    }
}

/* ================= MLP ================= */
typedef struct { int nin,hid,nout; P W1,b1,W2,b2; float *h; float logits[32]; } MLP;
static void mlp_init(MLP*m,int nin,int hid,int nout,float s){ m->nin=nin;m->hid=hid;m->nout=nout;
    pinit(&m->W1,hid*nin,s); pinit(&m->b1,hid,0.01f); pinit(&m->W2,nout*hid,s); pinit(&m->b2,nout,0.01f);
    m->h=malloc(4*hid); }
static float mlp_forward(MLP*m,const float*x,int label){
    for(int i=0;i<m->hid;i++){ float s=m->b1.w[i]; for(int j=0;j<m->nin;j++) s+=m->W1.w[i*m->nin+j]*x[j]; m->h[i]= s>0?s:0; }
    for(int o=0;o<m->nout;o++){ float s=m->b2.w[o]; for(int i=0;i<m->hid;i++) s+=m->W2.w[o*m->hid+i]*m->h[i]; m->logits[o]=s; }
    float pr[32]; softmax(m->logits,pr,m->nout); return -logf(pr[label]+1e-9f);
}
static void mlp_backward(MLP*m,const float*x,int label){
    float pr[32]; softmax(m->logits,pr,m->nout);
    float dl[32]; for(int o=0;o<m->nout;o++) dl[o]=pr[o]-(o==label?1.f:0.f);
    float dh[512]; for(int i=0;i<m->hid;i++) dh[i]=0;
    for(int o=0;o<m->nout;o++){ m->b2.gr[o]+=dl[o]; for(int i=0;i<m->hid;i++){ m->W2.gr[o*m->hid+i]+=dl[o]*m->h[i]; dh[i]+=m->W2.w[o*m->hid+i]*dl[o]; } }
    for(int i=0;i<m->hid;i++) if(m->h[i]<=0) dh[i]=0;
    for(int i=0;i<m->hid;i++){ m->b1.gr[i]+=dh[i]; for(int j=0;j<m->nin;j++) m->W1.gr[i*m->nin+j]+=dh[i]*x[j]; }
}

/* ================= tasks ================= */
static int TDIN(int task){ return task==0?2: task==1?7: task==2?2: 28; }
static int TOUT(int task){ return task==0?2: task==1?7: task==2?6: 10; }

static void gen_task(int task,int T,int N,float*X,int*Y){
    for(int s=0;s<N;s++){
        int st=0; int cur[3]={0,1,2};
        int A[3]={1,0,2}, B[3]={0,2,1};
        for(int t=0;t<T;t++){
            int din=TDIN(task); float*row=X+((long)s*T+t)*din;
            memset(row,0,4*din);
            if(task==0){ int b=urand()<0.5f?0:1; row[b]=1; st^=b; }
            else if(task==1){ int a=(int)(urand()*7); row[a]=1; st=(st+a)%7; }
            else if(task==2){ int gi=urand()<0.5f?0:1; row[gi]=1;
                int*g= gi?B:A; int r[3]; for(int i=0;i<3;i++) r[i]=g[cur[i]]; for(int i=0;i<3;i++) cur[i]=r[i]; }
        }
        if(task<2) Y[s]=st;
        else if(task==2){ int P[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
            for(int c=0;c<6;c++) if(P[c][0]==cur[0]&&P[c][1]==cur[1]&&P[c][2]==cur[2]){ Y[s]=c; break; } }
    }
}

/* ================= MNIST ================= */
static uint32_t be32(const unsigned char*b){ return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3]; }
static float* read_images(const char*path,int*want,int*outN){
    FILE*f=fopen(path,"rb"); if(!f){ printf("cannot open %s\n",path); *outN=0; return NULL; }
    unsigned char hdr[16]; fread(hdr,1,16,f); int N=be32(hdr+4);
    if(*want>0 && *want<N) N=*want;
    float*X=malloc(4*(long)N*784); unsigned char*row=malloc(784);
    for(int i=0;i<N;i++){ fread(row,1,784,f); for(int j=0;j<784;j++) X[(long)i*784+j]=row[j]/255.0f; }
    free(row); fclose(f); *outN=N; return X;
}
static unsigned char* read_labels(const char*path,int N){
    FILE*f=fopen(path,"rb"); if(!f) return NULL; unsigned char hdr[8]; fread(hdr,1,8,f);
    unsigned char*Y=malloc(N); fread(Y,1,N,f); fclose(f); return Y;
}

/* ================= train/eval ================= */
static long count_params(GRNN*m){ return m->Wx.n+m->Wo.n+m->bo.n; }
static long count_params_mlp(MLP*m){ return m->W1.n+m->b1.n+m->W2.n+m->b2.n; }

static float train_grnn(int task,Group G,int T,int N,int epochs){
    GRNN m; grnn_init(&m,G,TDIN(task),TOUT(task), task==3?0.02f:0.3f);
    float*X=malloc(4*(long)N*T*TDIN(task)); int*Y=malloc(4*N); gen_task(task,T,N,X,Y);
    int B=32;
    for(int ep=0;ep<epochs;ep++){ double epL=0; long epC=0;
        for(int s=0;s<N;s+=B){ pzero(&m.Wx);pzero(&m.Wo);pzero(&m.bo);
            int b2=s+B<N?s+B:N;
            for(int i=s;i<b2;i++){ epL+=grnn_forward(&m,T,X+(long)i*T*TDIN(task),Y[i]); epC++; grnn_backward(&m,Y[i]); }
            if(ep==0&&s==0){ double gn=0,wl=0; for(int i=0;i<m.Wx.n;i++){ gn+=fabs(m.Wx.gr[i]); wl+=fabs(m.Wx.w[i]); }
                double dq0=0; for(int i=0;i<m.Wo.n;i++) dq0+=fabs(m.Wo.gr[i]);
                printf("      [dbg] gradL1 Wx=%.3e Wo=%.3e  WxL1=%.3e\n",gn,dq0,wl); }
            ADAM_T++; padam(&m.Wx,0.01f); padam(&m.Wo,0.01f); padam(&m.bo,0.01f); }
        printf("      ep%2d loss %.4f\n",ep,epL/epC);
    }
    /* eval at several lengths */
    printf("    GRNN(%d params): ",(int)count_params(&m));
    for(int T2=12;T2<=96;T2*=2){
        int NE=1000; float*XE=malloc(4*(long)NE*T2*TDIN(task)); int*YE=malloc(4*NE); gen_task(task,T2,NE,XE,YE);
        int ok=0; for(int i=0;i<NE;i++){ grnn_forward(&m,T2,XE+(long)i*T2*TDIN(task),YE[i]);
            int am=0; for(int o=1;o<m.nout;o++) if(m.logits[o]>m.logits[am])am=o; if(am==YE[i])ok++; }
        printf("T=%d:%.2f ",T2,(float)ok/NE); free(XE);free(YE);
    }
    printf("\n"); free(X);free(Y); return 0;
}
static float train_mlp(int task,int T,int N,int epochs){
    int nin=T*TDIN(task); MLP m; mlp_init(&m,nin,256,TOUT(task),0.05f);
    float*X=malloc(4*(long)N*nin); int*Y=malloc(4*N); gen_task(task,T,N,X,Y);
    int B=32;
    for(int ep=0;ep<epochs;ep++){ double epL=0; long epC=0;
        for(int s=0;s<N;s+=B){ pzero(&m.W1);pzero(&m.b1);pzero(&m.W2);pzero(&m.b2);
            int b2=s+B<N?s+B:N;
            for(int i=s;i<b2;i++){ epL+=mlp_forward(&m,X+(long)i*nin,Y[i]); epC++; mlp_backward(&m,X+(long)i*nin,Y[i]); }
            ADAM_T++; padam(&m.W1,0.002f); padam(&m.b1,0.002f); padam(&m.W2,0.002f); padam(&m.b2,0.002f); }
        printf("      ep%2d loss %.4f\n",ep,epL/epC);
    }
    int NE=1000; float*XE=malloc(4*(long)NE*nin); int*YE=malloc(4*NE); gen_task(task,T,NE,XE,YE);
    int ok=0; for(int i=0;i<NE;i++){ mlp_forward(&m,XE+(long)i*nin,YE[i]);
        int am=0; for(int o=1;o<m.nout;o++) if(m.logits[o]>m.logits[am])am=o; if(am==YE[i])ok++; }
    printf("    MLP (%d params, fixed T=%d): %.2f\n",(int)count_params_mlp(&m),T,(float)ok/NE);
    free(X);free(Y);free(XE);free(YE); return 0;
}

int main(int argc,char**argv){
    const char*mdir = argc>1?argv[1]:"/tmp/opencode/mnist";
    printf("=== algorithmic Whorf: group-valued RNN vs MLP (C) ===\n\n");
    struct { int task; const char*name; Group G; } tasks[3];
    tasks[0]=(typeof(tasks[0])){0,"parity (Z/2)",mk_cyclic(2)};
    tasks[1]=(typeof(tasks[1])){1,"sum mod 7 (Z/7)",mk_cyclic(7)};
    tasks[2]=(typeof(tasks[2])){2,"S3 product (non-abelian)",mk_sym3()};
    for(int i=0;i<3;i++){
        printf("[%s]  train T=12, N=6000\n",tasks[i].name);
        train_grnn(tasks[i].task,tasks[i].G,12,6000,12);
        train_mlp(tasks[i].task,12,6000,12);
        printf("\n");
    }
    /* ---- MNIST control ---- */
    char p1[512],p2[512],p3[512],p4[512];
    snprintf(p1,512,"%s/train-images-idx3-ubyte",mdir); snprintf(p2,512,"%s/train-labels-idx1-ubyte",mdir);
    snprintf(p3,512,"%s/t10k-images-idx3-ubyte",mdir); snprintf(p4,512,"%s/t10k-labels-idx1-ubyte",mdir);
    int NT=12000, NE=2000;
    float*XT=read_images(p1,&NT,&NT); unsigned char*YT=read_labels(p2,NT);
    float*XE=read_images(p3,&NE,&NE); unsigned char*YE=read_labels(p4,NE);
    if(XT&&XE){
        int T=28,din=28;
        printf("[MNIST control]  train N=%d, T=28 rows\n",NT);
        GRNN g; grnn_init(&g,mk_cyclic(8),din,10,0.02f);
        for(int ep=0;ep<6;ep++)
            for(int s=0;s<NT;s+=64){ pzero(&g.Wx);pzero(&g.Wo);pzero(&g.bo);
                int b2=s+64<NT?s+64:NT;
                for(int i=s;i<b2;i++){ grnn_forward(&g,T,XT+(long)i*784,YT[i]); grnn_backward(&g,YT[i]); }
                ADAM_T++; padam(&g.Wx,0.01f);padam(&g.Wo,0.01f);padam(&g.bo,0.01f); }
        int ok=0; for(int i=0;i<NE;i++){ grnn_forward(&g,T,XE+(long)i*784,YE[i]);
            int am=0; for(int o=1;o<10;o++) if(g.logits[o]>g.logits[am])am=o; if(am==YE[i])ok++; }
        printf("    GRNN (order-8 group state, %d params): test acc = %.3f\n",(int)count_params(&g),(float)ok/NE);

        MLP ml; mlp_init(&ml,784,128,10,0.02f);
        for(int ep=0;ep<6;ep++)
            for(int s=0;s<NT;s+=64){ pzero(&ml.W1);pzero(&ml.b1);pzero(&ml.W2);pzero(&ml.b2);
                int b2=s+64<NT?s+64:NT;
                for(int i=s;i<b2;i++){ mlp_forward(&ml,XT+(long)i*784,YT[i]); mlp_backward(&ml,XT+(long)i*784,YT[i]); }
                ADAM_T++; padam(&ml.W1,0.01f);padam(&ml.b1,0.01f);padam(&ml.W2,0.01f);padam(&ml.b2,0.01f); }
        ok=0; for(int i=0;i<NE;i++){ mlp_forward(&ml,XE+(long)i*784,YE[i]);
            int am=0; for(int o=1;o<10;o++) if(ml.logits[o]>ml.logits[am])am=o; if(am==YE[i])ok++; }
        printf("    MLP (784-128-10, %d params):        test acc = %.3f\n",(int)count_params_mlp(&ml),(float)ok/NE);
    }
    printf("\ndone.\n");
    return 0;
}
